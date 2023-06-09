import os

import luigi
import yaml

from morgoth.auto_loc.time_selection import TimeSelection, TimeSelectionBB
from morgoth.downloaders import GatherTrigdatDownload, DownloadTrigdat

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelectionHandler(luigi.Task):
    grb_name = luigi.Parameter()
    version = luigi.Parameter()
    report_type = luigi.Parameter()

    def requires(self):
        # return GatherTrigdatDownload(grb_name=self.grb_name)
        if self.report_type == "trigdat":
            return DownloadTrigdat(grb_name =self.grb_name, version = self.version)


    def output(self):
        filename = f"time_selection_{self.version}.yml"

        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type ,filename))

    def run(self):
        """
        with self.input().open() as f:
            trigdat_version = yaml.safe_load(f)["trigdat_version"]
        """
        trigdat_versions = ["v02","v01","v00"]
        if self.report_type == "trigdat":
            tf_name = f"glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit"

            trigdat_file = os.path.join(base_dir,self.grb_name,"trigdat",tf_name)

            time_selection = TimeSelectionBB(
                grb_name=self.grb_name, trigdat_file=trigdat_file, fine=True
            )

            time_selection.save_yaml(
                os.path.join(base_dir, self.grb_name,self.report_type,f"time_selection_{self.version}.yml")
            )
        elif self.report_type == "tte":
            for tv in trigdat_versions:
                try:
                    with open(os.path.join(base_dir, self.grb_name,"trigdat",f"time_selection_{tv}.yml"),"r") as f:
                        con = yaml.safe_load(f)
                    with open(os.path.join(base_dir,self.grb_name,"tte",f"time_selection_{self.version}.yml"),"w+") as f:
                        yaml.dump(con,f)
                except FileNotFoundError:
                    pass
