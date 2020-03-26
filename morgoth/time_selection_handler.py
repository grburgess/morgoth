import luigi
import os

from morgoth.downloaders import DownloadTrigdat
from morgoth.auto_loc.time_selection import TimeSelection

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelectionHandler(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v01")  # TODO change this to v00 for not testing!!!!!!!!

    def requires(self):
        return DownloadTrigdat(grb_name=self.grb_name, version=self.version)

    def output(self):
        filename = "time_selection.yml"

        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        time_selection = TimeSelection(self.grb_name, self.version)

        filename = "time_selection.yml"

        tmp = os.path.join(base_dir, self.grb_name, filename)

        time_selection.save_yaml(tmp)
