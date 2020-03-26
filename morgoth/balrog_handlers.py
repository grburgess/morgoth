import luigi
from luigi.contrib.external_program import ExternalPythonProgramTask
import os

from morgoth.bkg_fit_handler import BackgroundFitTTE, BackgroundFitTrigdat
from morgoth.configuration import morgoth_config

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")

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

        filename = f"{self.report_type.lower()}_{self.version}_report.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"{self.report_type.lower()}_{self.version}_report.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class RunBalrogTTE(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):

        return BackgroundFitTTE(grb_name=self.grb_name, version=self.version)#[BackgroundFitTTE(grb_name=self.grb_name, version=self.version, detector=d) for d in _gbm_detectors]
        
    def output(self):
        
        filename = f"tte_{self.version}_loc_results.fits"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, "fit_results", filename))

    def run(self):

        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/auto_loc/fit_script.py"

        time_selection_file_path = os.path.join(base_dir, self.grb_name, "time_selection.yml")

        bkg_fit_file_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_tte_{self.version}.yml")
        
        os.system(f"mpiexec -n {n_cores_multinest} {path_to_python} {fit_script_path} {self.grb_name} {self.version} {bkg_fit_file_path} {time_selection_file_path} tte")


class RunBalrogTrigdat(luigi.ExternalTask):# ExternalPythonProgramTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    #def program_args(self):
    #    return ['./auto_loc/fit_script.py', n_cores_multinest, self.grb_name, self.version, bkg_fit_file_path, time_selection_file_path, 'trigdat', path_to_python]
    
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

