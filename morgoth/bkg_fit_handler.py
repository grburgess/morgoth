import os
import time

import luigi
import yaml

from morgoth.auto_loc.bkg_fit import BkgFittingTTE, BkgFittingTrigdat
from morgoth.downloaders import (
    DownloadCSPECFile,
    DownloadTTEFile,
    DownloadTrigdat,
    GatherTrigdatDownload,
)
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
    resources = {"max_workers": 1}
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "time_selection": TimeSelectionHandler(
                grb_name=self.grb_name, version=self.version, report_type="tte"
            ),
            "trigdat_version": GatherTrigdatDownload(grb_name=self.grb_name),
            "tte_files": [
                DownloadTTEFile(
                    grb_name=self.grb_name, version=self.version, detector=det
                )
                for det in _gbm_detectors
            ],
            "cspec_files": [
                DownloadCSPECFile(
                    grb_name=self.grb_name, version=self.version, detector=det
                )
                for det in _gbm_detectors
            ],
        }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, "tte", self.version)
        return {
            "bkg_fit_yml": luigi.LocalTarget(
                os.path.join(base_job, f"bkg_fit_tte_{self.version}.yml")
            ),
            "bkg_fit_files": [
                luigi.LocalTarget(
                    os.path.join(base_job, "bkg_files", f"bkg_det_{d}.h5")
                )
                for d in _gbm_detectors
            ],
        }

    def run(self):
        base_job = os.path.join(base_dir, self.grb_name, "tte", self.version)

        # Get the first trigdat version and gather the result of the background
        with self.input()["trigdat_version"].open() as f:
            trigdat_version = yaml.safe_load(f)["trigdat_version"]

        trigdat_file = DownloadTrigdat(
            grb_name=self.grb_name, version=trigdat_version
        ).output()
        trigdat_bkg = BackgroundFitTrigdat(
            grb_name=self.grb_name, version=trigdat_version
        ).output()

        # Fit TTE background
        bkg_fit = BkgFittingTTE(
            self.grb_name,
            self.version,
            trigdat_file=trigdat_file.path,
            tte_files=self.input()["tte_files"],
            cspec_files=self.input()["cspec_files"],
            time_selection_file_path=self.input()["time_selection"].path,
            bkg_fitting_file_path=trigdat_bkg["bkg_fit_yml"].path,
        )

        # Save background fit
        bkg_fit.save_bkg_file(os.path.join(base_job, "bkg_files"))

        # Save lightcurves
        bkg_fit.save_lightcurves(os.path.join(base_job, "plots", "lightcurves"))

        # Save background fit yaml
        bkg_fit.save_yaml(self.output()["bkg_fit_yml"].path)


class BackgroundFitTrigdat(luigi.Task):
    resources = {"max_workers": 1}
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "trigdat_file": DownloadTrigdat(
                grb_name=self.grb_name, version=self.version
            ),
            "time_selection": TimeSelectionHandler(
                grb_name=self.grb_name, version=self.version, report_type="trigdat"
            ),
        }

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, "trigdat", self.version)
        return {
            "bkg_fit_yml": luigi.LocalTarget(
                os.path.join(base_job, f"bkg_fit_trigdat_{self.version}.yml")
            ),
            "bkg_fit_files": [
                luigi.LocalTarget(
                    os.path.join(base_job, "bkg_files", f"bkg_det_{d}.h5")
                )
                for d in _gbm_detectors
            ],
        }

    def run(self):
        base_job = os.path.join(base_dir, self.grb_name, "trigdat", self.version)

        # Fit Trigdat background
        bkg_fit = BkgFittingTrigdat(
            self.grb_name,
            self.version,
            trigdat_file=self.input()["trigdat_file"].path,
            time_selection_file_path=self.input()["time_selection"].path,
        )

        # Save background fit
        bkg_fit.save_bkg_file(os.path.join(base_job, "bkg_files"))

        # Save lightcurves
        bkg_fit.save_lightcurves(os.path.join(base_job, "plots", "lightcurves"))

        # Save background fit yaml
        bkg_fit.save_yaml(self.output()["bkg_fit_yml"].path)
