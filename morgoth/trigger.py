import os
import re
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from lxml import etree
import luigi
import numpy as np
import yaml

from morgoth.utils.env import get_env_value
from morgoth.utils.file_utils import if_directory_not_existing_then_make

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


_gbm_detectors = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
)


base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


# most_likely_lookup = {
#     "4": "GRB",
#     "5": "GENERIC_SGR",
#     "6": " GENERIC_TRANSIENT",
#     "7": "DISTANT_PARTICLES",
#     "10": "SGR_1806_20",
#     "11": "GROJ_0422_32",
# }


most_likely_lookup = {
    "0": "ERROR",
    "1": "UNRELIABLE_LOCATION",
    "2": "LOCAL_PARTICLES",
    "3": "BELOW_HORIZON",
    "4": "GRB",
    "5": "GENERIC_SGR",
    "6": "GENERIC_TRANSIENT",
    "7": "DISTANT_PARTICLES",
    "8": "SOLAR_FLARE",
    "9": "CYG_X1",
    "10": "SGR_1806_20 ",
    "11": "GROJ_0422_32",
    "12": "undefined",
    "13": "undefined",
    "14": "undefined",
    "15": "undefined",
    "16": "undefined",
    "17": "undefined",
    "18": "undefined",
    "19": "TGF",
}


def parse_trigger_file_and_write(root, payload):

    tmp = root.find(".//{*}ISOTime").text
    yy, mm, dd = re.match(
        r"^\d{2}(\d{2})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}\.\d{2}$", tmp
    ).groups()

    trigger_day_start = f"20{yy}-{mm}-{dd}"

    time_frac = (
        datetime.strptime(tmp, "%Y-%m-%dT%H:%M:%S.%f")
        - datetime.strptime(trigger_day_start, "%Y-%m-%d")
    ).total_seconds() / timedelta(1).total_seconds()

    frac = str(int(np.round(time_frac * 1000)))

    if len(frac) == 1:
        frac = f"00{frac}"
    elif len(frac) == 2:
        frac = f"0{frac}"

    burst_name = f"GRB{yy}{mm}{dd}{frac}"

    burst_number = f"{yy}{mm}{dd}{frac}"

    pos2d = root.find(".//{*}Position2D")
    ra = float(pos2d.find(".//{*}C1").text)
    dec = float(pos2d.find(".//{*}C2").text)
    radius = float(pos2d.find(".//{*}Error2Radius").text)

    # alert_type = int(root.find(".//Param[@name='Packet_Type']").attrib["value"])
    tmp = str(root.find(".//Param[@name='Most_Likely_Index']").attrib["value"])
    most_likely = most_likely_lookup[tmp]
    most_likely_prob = float(
        root.find(".//Param[@name='Most_Likely_Prob']").attrib["value"]
    )

    tmp = str(root.find(".//Param[@name='Sec_Most_Likely_Index']").attrib["value"])
    most_likely_2 = most_likely_lookup[tmp]
    most_likely_prob_2 = float(
        root.find(".//Param[@name='Sec_Most_Likely_Prob']").attrib["value"]
    )

    lc_file = root.find(".//Param[@name='LightCurve_URL']").attrib["value"]

    # now we want to store the folder directory

    main_ftp_directory = os.path.join("/".join(lc_file.split("/")[:-2]), "current")

    if "https" not in main_ftp_directory:

        main_ftp_directory = main_ftp_directory.replace("http", "https")

    uri = main_ftp_directory

    out_file_writer = GBMTriggerFile(
        name=burst_name,
        burst_number=burst_number,
        ra=ra,
        dec=dec,
        radius=radius,
        uri=uri,
        most_likely=most_likely,
        most_likely_prob=most_likely_prob,
        most_likely_2=most_likely_2,
        most_likely_prob_2=most_likely_prob_2,
    )

    directory = os.path.join(base_dir, burst_name)
    if_directory_not_existing_then_make(directory)

    out_file_writer.write(os.path.join(directory, "grb_parameters.yml"))

    # now make a file that will tell luigi to run

    # os.system(f"touch {os.path.join(directory,'ready')}")

    # now save the xml_file

    #tree = etree.XML(payload)
    
    with open(os.path.join(directory, "gbm_flight_voe.xml"), "w") as f:

        
        
        f.write(payload)

    
    return burst_name


class GBMTriggerFile(object):
    def __init__(
        self,
        name,
        burst_number,
        ra,
        dec,
        radius,
        uri,
        most_likely,
        most_likely_prob,
        most_likely_2,
        most_likely_prob_2,
    ):

        self._params = dict(
            name=name,
            burst_number=burst_number,
            ra=ra,
            dec=dec,
            radius=radius,
            uri=uri,
            most_likely=most_likely,
            most_likely_prob=most_likely_prob,
            most_likely_2=most_likely_2,
            most_likely_prob_2=most_likely_prob_2,
        )

        self.ra = ra
        self.dec = dec
        self.name = name
        self.burst_number = burst_number
        self.radius = radius
        self.uri = uri
        self.most_likely = most_likely
        self.most_likely_prob = most_likely_prob
        self.most_likely_2 = most_likely_2
        self.most_likely_prob_2 = most_likely_prob_2

    def write(self, file_name):

        with open(file_name, "w") as f:

            yaml.dump(self._params, f, Dumper=yaml.SafeDumper, default_flow_style=False)

    @classmethod
    def from_file(cls, file_name):

        with file_name.open("r") as f:

            stuff = yaml.load(f, Loader=yaml.SafeLoader)

        return cls(
            name=stuff["name"],
            burst_number=stuff["burst_number"],
            ra=stuff["ra"],
            dec=stuff["dec"],
            radius=stuff["radius"],
            uri=stuff["uri"],
            most_likely=stuff["most_likely"],
            most_likely_prob=stuff["most_likely_prob"],
            most_likely_2=stuff["most_likely_2"],
            most_likely_prob_2=stuff["most_likely_prob_2"],
        )


class OpenGBMFile(luigi.Task):

    grb = luigi.Parameter()

    def requires(self):

        return None

    def output(self):
        return luigi.LocalTarget(os.path.join(base_dir, self.grb, "grb_parameters.yml"))
