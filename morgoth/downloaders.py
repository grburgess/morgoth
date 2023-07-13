import os
import time

import luigi
import yaml

from morgoth.configuration import morgoth_config
from morgoth.trigger import GBMTriggerFile, OpenGBMFile
from morgoth.utils import file_utils
from morgoth.utils.download_file import BackgroundDownload
from morgoth.utils.env import get_env_value

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class GatherTrigdatDownload(luigi.Task):
    grb_name = luigi.Parameter()

    def requires(self):
        return OpenGBMFile(grb=self.grb_name)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(base_dir, self.grb_name, f"gather_trigdat_complete.yml")
        )

    def run(self):
        # the time spent waiting so far
        time_spent = 0  # seconds

        while True:
            if DownloadTrigdat(grb_name=self.grb_name, version="v00").complete():
                version = "v00"
                break

            if DownloadTrigdat(grb_name=self.grb_name, version="v01").complete():
                version = "v01"
                break

            if DownloadTrigdat(grb_name=self.grb_name, version="v02").complete():
                version = "v02"
                break

            time.sleep(2)

            # up date the time we have left
            time_spent += 2

        print(f"The total waiting time was: {time_spent}")

        version_dict = {"trigdat_version": version}

        with self.output().open("w") as f:
            yaml.dump(version_dict, f, Dumper=yaml.SafeDumper, default_flow_style=False)


class DownloadTrigdat(luigi.Task):
    """
    Downloads a Trigdat file of a given
    version
    """

    priority = 100
    grb_name = luigi.Parameter()
    version = luigi.Parameter()

    def requires(self):
        return OpenGBMFile(grb=self.grb_name)

    def output(self):
        trigdat = f"glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit"
        return luigi.LocalTarget(
            os.path.join(base_dir, self.grb_name, "trigdat", trigdat)
        )

    def run(self):
        # get the info from the stored yaml file
        info = GBMTriggerFile.from_file(self.input())

        # parse the trigdat
        trigdat = f"glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit"

        uri = os.path.join(info.uri, trigdat)

        store_path = os.path.join(base_dir, info.name, "trigdat")
        dl = BackgroundDownload(
            uri,
            store_path,
            wait_time=float(
                morgoth_config["download"]["trigdat"][self.version]["interval"]
            ),
            max_time=float(
                morgoth_config["download"]["trigdat"][self.version]["max_time"]
            ),
        )
        dl.run()

        # Create the version subfolder when download is done
        file_utils.if_directory_not_existing_then_make(
            os.path.join(base_dir, info.name, "trigdat", self.version)
        )


class DownloadTTEFile(luigi.Task):
    priority = -100
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")
    detector = luigi.Parameter()

    def requires(self):
        return OpenGBMFile(grb=self.grb_name)

    def output(self):
        tte = f"glg_tte_{self.detector}_bn{self.grb_name[3:]}_{self.version}.fit"
        return luigi.LocalTarget(
            os.path.join(base_dir, self.grb_name, "tte", "data", tte)
        )

    def run(self):
        info = GBMTriggerFile.from_file(self.input())

        print(info)

        tte = f"glg_tte_{self.detector}_bn{self.grb_name[3:]}_{self.version}.fit"
        uri = os.path.join(info.uri, tte)
        print(uri)

        store_path = os.path.join(base_dir, info.name, "tte", "data")
        dl = BackgroundDownload(
            uri,
            store_path,
            wait_time=float(
                morgoth_config["download"]["tte"][self.version]["interval"]
            ),
            max_time=float(morgoth_config["download"]["tte"][self.version]["max_time"]),
        )
        dl.run()

        # Create the version subfolder when download is done
        file_utils.if_directory_not_existing_then_make(
            os.path.join(base_dir, info.name, "tte", self.version)
        )


class DownloadCSPECFile(luigi.Task):
    priority = -100
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v01")
    detector = luigi.Parameter()

    def requires(self):
        return OpenGBMFile(grb=self.grb_name)

    def output(self):
        cspec = f"glg_cspec_{self.detector}_bn{self.grb_name[3:]}_{self.version}.pha"
        return luigi.LocalTarget(
            os.path.join(base_dir, self.grb_name, "tte", "data", cspec)
        )

    def run(self):
        info = GBMTriggerFile.from_file(self.input())

        print(info)

        cspec = f"glg_cspec_{self.detector}_bn{self.grb_name[3:]}_{self.version}.pha"

        uri = os.path.join(info.uri, cspec)
        print(uri)

        store_path = os.path.join(base_dir, info.name, "tte", "data")
        dl = BackgroundDownload(
            uri,
            store_path,
            wait_time=float(
                morgoth_config["download"]["cspec"][self.version]["interval"]
            ),
            max_time=float(
                morgoth_config["download"]["cspec"][self.version]["max_time"]
            ),
        )
        dl.run()
