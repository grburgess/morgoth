import os

import luigi
import yaml

from morgoth.auto_loc.time_selection import TimeSelection, TimeSelectionBB
from morgoth.downloaders import GatherTrigdatDownload, DownloadTrigdat

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class TimeSelectionHandler(luigi.Task):
    grb_name = luigi.Parameter()

    def requires(self):
        return GatherTrigdatDownload(grb_name=self.grb_name)

    def output(self):
        filename = "time_selection.yml"

        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        with self.input().open() as f:
            trigdat_version = yaml.safe_load(f)["trigdat_version"]

        trigdat_file = DownloadTrigdat(
            grb_name=self.grb_name, version=trigdat_version
        ).output()

        time_selection = TimeSelectionBB(
            grb_name=self.grb_name, trigdat_file=trigdat_file.path, fine=True
        )

        time_selection.save_yaml(
            os.path.join(base_dir, self.grb_name, "time_selection.yml")
        )
