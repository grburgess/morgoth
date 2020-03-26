import luigi
import os
import shutil

from morgoth.utils.file_utils import if_directory_not_existing_then_make
from morgoth.utils.package_data import get_path_of_data_file
from morgoth.utils.env import get_env_value
from morgoth.utils.result_reader import ResultReader
from morgoth.bkg_fit_handler import BackgroundFitTTE, BackgroundFitTrigdat
from morgoth.configuration import morgoth_config

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")
n_cores_multinest = morgoth_config["multinest"]["n_cores"]
path_to_python = morgoth_config["multinest"]["path_to_python"]


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
        return BackgroundFitTTE(grb_name=self.grb_name, version=self.version)
        
    def output(self):
        
        filename = f"tte_{self.version}_loc_results.fits"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, "fit_results", filename))

    def run(self):
        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/auto_loc/fit_script.py"

        time_selection_file_path = os.path.join(base_dir, self.grb_name, "time_selection.yml")

        bkg_fit_file_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_tte_{self.version}.yml")
        
        os.system(f"mpiexec -n {n_cores_multinest} {path_to_python} {fit_script_path} {self.grb_name} {self.version} {bkg_fit_file_path} {time_selection_file_path} tte")


class RunBalrogTrigdat(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")
    
    def requires(self):
        return BackgroundFitTrigdat(grb_name=self.grb_name, version=self.version)

    def output(self):
        filename = f"trigdat_{self.version}_loc_results.fits"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, "fit_results",filename))

    def run(self):

        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/auto_loc/fit_script.py"

        time_selection_file_path = os.path.join(base_dir, self.grb_name, "time_selection.yml")

        bkg_fit_file_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_trigdat_{self.version}.yml")
        
        os.system(f"mpiexec -n {n_cores_multinest} {path_to_python} {fit_script_path} {self.grb_name} {self.version} {bkg_fit_file_path} {time_selection_file_path} trigdat")


