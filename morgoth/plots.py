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


class CreateAllLightcurves(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "n0": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n0", version=self.version),
            "n1": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n1", version=self.version),
            "n2": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n2", version=self.version),
            "n3": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n3", version=self.version),
            "n4": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n4", version=self.version),
            "n5": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n5", version=self.version),
            "n6": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n6", version=self.version),
            "n7": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n7", version=self.version),
            "n8": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n8", version=self.version),
            "n9": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n9", version=self.version),
            "na": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="na", version=self.version),
            "nb": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="nb", version=self.version),
            "b0": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b0", version=self.version),
            "b1": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b1", version=self.version),
            "b2": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b2", version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")