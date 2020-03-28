import luigi
import os

from morgoth.downloaders import DownloadTTEFile, DownloadTrigdat, DownloadCSPECFile
from morgoth.time_selection_handler import TimeSelectionHandler

from morgoth.auto_loc.bkg_fit import BkgFittingTrigdat, BkgFittingTTE

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


class BackgroundFitTTE(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
            'trigdat_bkg': BackgroundFitTrigdat(grb_name=self.grb_name, version="v01"),  # CHANGE THIS TO v00
            'tte_file': [DownloadTTEFile(grb_name=self.grb_name,
                                         version=self.version,
                                         detector=d) for d in _gbm_detectors],
            'cspec_file': [DownloadCSPECFile(grb_name=self.grb_name,
                                             version=self.version,
                                             detector=d) for d in _gbm_detectors]
        }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, 'trigdat', self.version)
        return {
            'bkg_fit_yml': luigi.LocalTarget(os.path.join(base_job, f"bkg_fit_tte_{self.version}.yml")),
            'bkg_fit_files': [luigi.LocalTarget(os.path.join(base_job, 'bkg_files', f'bkg_det_{d}.h5'))
                              for d in _gbm_detectors],
        }

    def run(self):
        base_job = os.path.join(base_dir, self.grb_name, 'trigdat', self.version)

        # Fit Background
        bkg_fit = BkgFittingTTE(
            self.grb_name,
            self.version,
            time_selection_file_path=self.input()['time_selection'].path,
            bkg_fitting_file_path=self.input()['trigdat_bkg'].path
        )

        # Save background fit
        bkg_fit.save_bkg_file(
            os.path.join(base_job, "bkg_files")
        )

        # Save lightcurves
        bkg_fit.save_lightcurves(
            os.path.join(base_job, 'plots', 'lightcurves')
        )

        # Save background fit yaml
        bkg_fit.save_yaml(
            self.output()['bkg_fit_yml'].path
        )


class BackgroundFitTrigdat(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'trigdat_file': DownloadTrigdat(grb_name=self.grb_name, version=self.version),
            'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
        }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, 'trigdat', self.version)
        return {
            'bkg_fit_yml': luigi.LocalTarget(os.path.join(base_job, f"bkg_fit_trigdat_{self.version}.yml")),
            'bkg_fit_files': [luigi.LocalTarget(os.path.join(base_job, 'bkg_files', f'bkg_det_{d}.h5'))
                              for d in _gbm_detectors],
        }

    def run(self):
        base_job = os.path.join(base_dir, self.grb_name, 'trigdat', self.version)

        # Fit Background
        bkg_fit = BkgFittingTrigdat(
            self.grb_name,
            self.version,
            trigdat_file=self.input()['trigdat_file'].path,
            time_selection_file_path=self.input()['time_selection'].path
        )

        # Save background fit
        bkg_fit.save_bkg_file(
            os.path.join(base_job, "bkg_files")
        )

        # Save lightcurves
        bkg_fit.save_lightcurves(
            os.path.join(base_job, 'plots', 'lightcurves')
        )

        # Save background fit yaml
        bkg_fit.save_yaml(
            self.output()['bkg_fit_yml'].path
        )
