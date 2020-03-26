import luigi
import os
import yaml
from datetime import datetime
import shutil

from morgoth.utils.file_utils import if_directory_not_existing_then_make
from morgoth.utils.package_data import get_path_of_data_file
from morgoth.utils.download_file import BackgroundDownload
from morgoth.trigger import OpenGBMFile, GBMTriggerFile
from morgoth.configuration import morgoth_config
from morgoth.utils.env import get_env_value
from morgoth.downloaders import DownloadTTEFile, DownloadTrigdat
from morgoth.utils.result_reader import ResultReader

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
        base_job = os.path.join(base_dir, self.grb_name, self.report_type, self.version)
        result_name = f"{self.report_type}_{self.version}_fit_result.yml"

        return {
            'result': luigi.LocalTarget(os.path.join(base_job, result_name)),
            'post_equal_weights': luigi.LocalTarget(os.path.join(base_job, 'chains', 'post_equal_weights.dat'))
        }

    def run(self):
        out_dir = os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'chains')
        if_directory_not_existing_then_make(out_dir)
        shutil.copyfile(get_path_of_data_file('post_equal_weights.dat'), os.path.join(out_dir, 'post_equal_weights.dat'))

        result_path = f"{base_dir}/{self.grb_name}/{self.report_type}/{self.version}/fit_result/" \
            f"{self.grb_name}_{self.report_type}_{self.version}_loc_results.fits"

        result_reader = ResultReader(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            result_file=result_path,
            trigger_file='dummy'
        )

        filename = f"{self.report_type}_{self.version}_fit_result.yml"
        file_path = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        result_reader.save_result_yml(file_path)


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
