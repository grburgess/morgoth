import os

import luigi

from morgoth.auto_loc.time_selection import TimeSelection
from morgoth.downloaders import DownloadTrigdat

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelectionHandler(luigi.Task):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(
        default="v01"
    )  # TODO change this to v00 for not testing!!!!!!!!

    def requires(self):
        return DownloadTrigdat(grb_name=self.grb_name, version=self.version)

    def output(self):
        filename = "time_selection.yml"

        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        time_selection = TimeSelection(
            grb_name=self.grb_name, version=self.version, trigdat_file=self.input().path
        )

        time_selection.save_yaml(
            os.path.join(base_dir, self.grb_name, "time_selection.yml")
        )
