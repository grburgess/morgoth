import luigi
import os
from morgoth.utils.download_file import BackgroundDownload
from morgoth.trigger import OpenGBMFile, GBMTriggerFile
from morgoth.configuration import morgoth_config
from morgoth.downloaders import DownloadTTEFile, DownloadTrigdat

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


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


class ProcessFitResults(luigi.Task):

    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        if self.report_type.lower() == "tte":

            return RunBalrogTTE(grb_name=self.grb_name)

        elif self.report_type.lower() == "trigdat":

            return RunBalrogTrigdat(grb_name=self.grb_name, version=self.version)

        else:

            return None

    def output(self):

        filename = f"{self.report_type}_{self.version}_report.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"{self.report_type}_{self.version}_report.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class RunBalrogTTE(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        return [
            DownloadTTEFile(grb_name=self.grb_name, version=self.version, detector=d)
            for d in _gbm_detectors
        ]

    def output(self):

        filename = f"tte_{self.version}_fit.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"tte_{self.version}_fit.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class RunBalrogTrigdat(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        return DownloadTrigdat(grb_name=self.grb_name, version=self.version)

    def output(self):

        filename = f"trigdat_{self.version}_fit.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"trigdat_{self.version}_fit.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")
