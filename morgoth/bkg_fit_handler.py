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

    # detector = luigi.Parameter()

    def requires(self):
        return {'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
                'trigdat_bkg': BackgroundFitTrigdat(grb_name=self.grb_name, version="v01"),  # CHANGE THIS TO v00
                'tte_file': [DownloadTTEFile(grb_name=self.grb_name,
                                             version=self.version,
                                             detector=d) for d in _gbm_detectors],
                'cspec_file': [DownloadCSPECFile(grb_name=self.grb_name,
                                                 version=self.version,
                                                 detector=d) for d in _gbm_detectors]}

    def output(self):
        yml_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_tte_{self.version}.yml")

        return luigi.LocalTarget(yml_path)

    def run(self):
        time_selection_filename = "time_selection.yml"

        time_selection_path = os.path.join(base_dir, self.grb_name, time_selection_filename)

        bkg_trigdat_filename = "bkg_fit_trigdat_v01.yml"  # TODO change this to v00

        bkg_trigdat_path = os.path.join(base_dir, self.grb_name, bkg_trigdat_filename)

        bkg_fit = BkgFittingTTE(self.grb_name, self.version, time_selection_path, bkg_trigdat_path)

        bkg_files_dir = os.path.join(base_dir, self.grb_name, f"bkg_files_tte_{self.version}")

        lightcurves_dir = os.path.join(base_dir, self.grb_name, f"lightcurves_tte_{self.version}")

        bkg_fit.save_bkg_file(bkg_files_dir)

        bkg_fit.save_lightcurves(lightcurves_dir)

        yml_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_tte_{self.version}.yml")

        bkg_fit.save_yaml(yml_path)


class BackgroundFitTrigdat(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    # detector = luigi.Parameter()
    def requires(self):
        return {'time_selection': TimeSelectionHandler(grb_name=self.grb_name),
                'data_file:': DownloadTrigdat(grb_name=self.grb_name,
                                              version=self.version)}

    def output(self):
        yml_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_trigdat_{self.version}.yml")

        return luigi.LocalTarget(yml_path)

    def run(self):
        time_selection_filename = "time_selection.yml"

        time_selection_path = os.path.join(base_dir, self.grb_name, time_selection_filename)

        bkg_fit = BkgFittingTrigdat(self.grb_name, self.version, time_selection_path)

        bkg_files_dir = os.path.join(base_dir, self.grb_name, f"bkg_files_trigdat_{self.version}")

        lightcurves_dir = os.path.join(base_dir, self.grb_name, f"lightcurves_trigdat_{self.version}")

        bkg_fit.save_bkg_file(bkg_files_dir)

        bkg_fit.save_lightcurves(lightcurves_dir)

        yml_path = os.path.join(base_dir, self.grb_name, f"bkg_fit_trigdat_{self.version}.yml")

        bkg_fit.save_yaml(yml_path)
