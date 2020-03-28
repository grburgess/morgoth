import os
import time

import luigi
import yaml

from morgoth.auto_loc.bkg_fit import BkgFittingTTE, BkgFittingTrigdat
from morgoth.downloaders import DownloadCSPECFile, DownloadTTEFile, DownloadTrigdat
from morgoth.time_selection_handler import TimeSelectionHandler

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


class BackgroundFitTTE(luigi.Task):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
            'trigdat_version': GatherTrigdatBackgroundFit(grb_name=self.grb_name),
            'tte_files': [DownloadTTEFile(grb_name=self.grb_name,
                                          version=self.version,
                                          detector=det) for det in _gbm_detectors],
            'cspec_files': [DownloadCSPECFile(grb_name=self.grb_name,
                                              version=self.version,
                                              detector=det) for det in _gbm_detectors]
        }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, 'tte', self.version)
        return {
            'bkg_fit_yml': luigi.LocalTarget(os.path.join(base_job, f"bkg_fit_tte_{self.version}.yml")),
            'bkg_fit_files': [luigi.LocalTarget(os.path.join(base_job, 'bkg_files', f'bkg_det_{d}.h5'))
                              for d in _gbm_detectors],
        }

    def run(self):
        base_job = os.path.join(base_dir, self.grb_name, 'tte', self.version)

        # Get the first trigdat version and gather the result of the background
        with self.input()['trigdat_version'].open() as f:
            trigdat_version = yaml.safe_load(f)['trigdat_version']

        trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=trigdat_version).output()
        trigdat_bkg = BackgroundFitTrigdat(grb_name=self.grb_name, version=trigdat_version).output()

        # Fit TTE background
        bkg_fit = BkgFittingTTE(
            self.grb_name,
            self.version,
            trigdat_file=trigdat_file.path,
            tte_files=self.input()['tte_files'],
            cspec_files=self.input()['cspec_files'],
            time_selection_file_path=self.input()['time_selection'].path,
            bkg_fitting_file_path=trigdat_bkg['bkg_fit_yml'].path
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


class GatherTrigdatBackgroundFit(luigi.Task):
    grb_name = luigi.Parameter()

    def requires(self):
        return {
            'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
        }

    def output(self):
        return  luigi.LocalTarget(os.path.join(base_dir, self.grb_name, f"gather_trigdat_complete.yml"))

    def run(self):
        # the time spent waiting so far
        time_spent = 0  # seconds

        while True:
            if BackgroundFitTrigdat(grb_name=self.grb_name, version="v00").complete():
                version = 'v00'
                break
            else:
                print('version 0 not complete')

            if BackgroundFitTrigdat(grb_name=self.grb_name, version="v01").complete():
                version = 'v01'
                break
            else:
                print('version 1 not complete')

            if BackgroundFitTrigdat(grb_name=self.grb_name, version="v02").complete():
                version = 'v02'
                break
            else:
                print('version 2 not complete')

            time.sleep(2)

            # up date the time we have left
            time_spent += 2

        print(f'The total waiting time was: {time_spent}')

        version_dict = {
            'trigdat_version': version
        }

        with self.output().open('w') as f:
            yaml.dump(version_dict, f, default_flow_style=False)


class BackgroundFitTrigdat(luigi.Task):
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

        # Fit Trigdat background
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
