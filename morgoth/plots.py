import luigi
import os
import yaml

from morgoth.utils.env import get_env_value
from morgoth.balrog_handlers import ProcessFitResults

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class CreateLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):

        filename = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"

        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename)

        os.system(f"touch {tmp}")
