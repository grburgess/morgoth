import luigi
import os

from morgoth.trigger import OpenGBMFile, GBMTriggerFile
from morgoth.configuration import morgoth_config
from morgoth.balrog_handlers import ProcessFitResults

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")



# work backwards for the moment


class CreateAllPages(luigi.WrapperTask):

    grb_name = luigi.Parameter()

    def requires(self):

        yield CreateReportTTE(grb_name=self.grb_name)
        yield CreateReportTrigdat(grb_name=self.grb_name, version="v00")
        yield CreateReportTrigdat(grb_name=self.grb_name, version="v01")
        yield CreateReportTrigdat(grb_name=self.grb_name, version="v02")


class CreateReportTTE(luigi.Task):

    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        return ProcessFitResults(grb_name=self.grb_name, report_type="tte")

    def output(self):

        filename = f"tte_{self.version}_webpage.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"tte_{self.version}_webpage.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class CreateReportTrigdat(luigi.Task):

    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        return ProcessFitResults(
            grb_name=self.grb_name, version=self.version, report_type="trigdat"
        )

    def output(self):

        filename = f"trigdat_{self.version}_webpage.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"trigdat_{self.version}_webpage.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


