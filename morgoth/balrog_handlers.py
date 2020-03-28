import luigi
from luigi.contrib.external_program import ExternalProgramTask
import os
import shutil

from morgoth.downloaders import DownloadTrigdat
from morgoth.time_selection_handler import TimeSelectionHandler
from morgoth.trigger import OpenGBMFile
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
            return {
                'gbm_file:': OpenGBMFile(grb=self.grb_name),
                'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
                'bkg_fit': BackgroundFitTTE(grb_name=self.grb_name, version=self.version),
                'balrog': RunBalrogTTE(grb_name=self.grb_name),
            }

        elif self.report_type.lower() == "trigdat":
            return {
                'gbm_file:': OpenGBMFile(grb=self.grb_name),
                'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
                'bkg_fit': BackgroundFitTrigdat(grb_name=self.grb_name, version=self.version),
                'balrog': RunBalrogTrigdat(grb_name=self.grb_name, version=self.version),
            }

        else:
            return None

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, self.report_type, self.version)
        result_name = f"{self.report_type}_{self.version}_fit_result.yml"

        return {
            'result_file': luigi.LocalTarget(os.path.join(base_job, result_name)),
            'post_equal_weights': self.input()['balrog']['post_equal_weights']
        }

    def run(self):

        result_reader = ResultReader(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            trigger_file=self.input()['gbm_file'].path,
            time_selection_file=self.input()['time_selection'].path,
            background_file=self.input()['bkg_fit']['bkg_fit_yml'].path,
            result_file=self.input()['balrog']['fit_result'].path,
        )

        result_reader.save_result_yml(self.output()['result'].path)


class RunBalrogTTE(ExternalProgramTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
                'bkg_fit': BackgroundFitTTE(grb_name=self.grb_name, version=self.version),
                'time_selection': TimeSelectionHandler(grb_name=self.grb_name)
            }
        
    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, 'tte', self.version)
        fit_result_name = f"tte_{self.version}_loc_results.fits"
        spectral_plot_name = f"{self.grb_name}_spectrum_plot_tte_{self.version}.png"

        return {
            'fit_result': luigi.LocalTarget(os.path.join(base_job, fit_result_name)),
            'post_equal_weights': luigi.LocalTarget(os.path.join(base_job, 'chains',
                                                                 f'tte_{self.version}_post_equal_weights.dat')),
            'spectral_plot': luigi.LocalTarget(os.path.join(base_job, 'plots', spectral_plot_name))
        }

    def program_args(self):
        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/auto_loc/fit_script.py"

        command = [
            "mpiexec",
            f"-n",
            f"{n_cores_multinest}",
            f"{path_to_python}",
            f"{fit_script_path}",
            f"{self.grb_name}",
            f"{self.version}",
            f"",  # Trigdat file
            f"{self.input()['bkg_fit']['bkg_fit_yml'].path}",
            f"{self.input()['time_selection'].path}",
            f"tte"
        ]
        return command


class RunBalrogTrigdat(ExternalProgramTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")
    
    def requires(self):
        return {
                'trigdat_file': DownloadTrigdat(grb_name=self.grb_name, version=self.version),
                'bkg_fit': BackgroundFitTrigdat(grb_name=self.grb_name, version=self.version),
                'time_selection': TimeSelectionHandler(grb_name=self.grb_name)
            }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, 'trigdat', self.version)
        fit_result_name = f"trigdat_{self.version}_loc_results.fits"
        spectral_plot_name = f"{self.grb_name}_spectrum_plot_trigdat_{self.version}.png"

        return {
            'fit_result': luigi.LocalTarget(os.path.join(base_job, fit_result_name)),
            'post_equal_weights': luigi.LocalTarget(os.path.join(base_job, 'chains',
                                                                 f'trigdat_{self.version}_post_equal_weights.dat')),
            'spectral_plot': luigi.LocalTarget(os.path.join(base_job, 'plots', spectral_plot_name))
        }

    def program_args(self):
        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/auto_loc/fit_script.py"

        command = [
            "mpiexec",
            f"-n",
            f"{n_cores_multinest}",
            f"{path_to_python}",
            f"{fit_script_path}",
            f"{self.grb_name}",
            f"{self.version}",
            f"{self.input()['trigdat_file'].path}",
            f"{self.input()['bkg_fit']['bkg_fit_yml'].path}",
            f"{self.input()['time_selection'].path}",
            f"trigdat"
        ]
        return command
