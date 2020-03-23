import luigi
import os

from morgoth.downloaders import DownloadTTEFile, DownloadTrigdat
from morgoth.time_selection_handler import TimeSelectionHandler

from auto_loc.fit.prepare_fit import BkgFittingTrigdat

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
    detector = luigi.Parameter()

    def requires(self):

        return {'Time_Selection': TimeSelectionHandler(grb_name=self.grb_name),
                'Data_File:': DownloadTTEFile(grb_name=self.grb_name,
                                              version=self.version,
                                              detector=self.detector)}
    
    def output(self):

        filename = f"tte_{self.version}_bkg_fit.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):

        filename = f"tte_{self.version}_bkg_fit.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")

    
class BackgroundFitTrigdat(luigi.ExternalTask):
    grb_name = luigi.Parameter()
    version = luigi.Parameter(default="v00")
    #detector = luigi.Parameter()
    def requires(self):

        return {'Time_Selection': TimeSelectionHandler(grb_name=self.grb_name),
                'Data_File:': DownloadTrigdat(grb_name=self.grb_name,
                                              version=self.version)}

    def output(self):
        bkg_files_dir = os.path.join(base_dir, self.grb_name, "bkg_files")

        lightcurves_dir = os.path.join(base_dir, self.grb_name, "lightcurves")
        
        return {"Bkg_fits": [luigi.LocalTarget(os.path.join(bkg_files_dir, f"bkg_det{d}.h5")) for d in _gbm_detectors],
                "Lightcurves": [luigi.LocalTarget(os.path.join(lightcurves_dir, f"{self.grb_name}_lightcurve_trigdat_detector_{d}_plot_{self.version}.png")) for d in _gbm_detectors]}

    def run(self):

        time_selection_filename = "time_selection.json"

        time_selection_path = os.path.join(base_dir, self.grb_name, time_selection_filename)
        
        bkg_fit = BkgFittingTrigdat(self.grb_name, self.version, time_selection_path)

        bkg_files_dir = os.path.join(base_dir, self.grb_name, "bkg_files")

        lightcurves_dir = os.path.join(base_dir, self.grb_name, "lightcurves")
        
        bkg_fit.save_bkg_file(bkg_files_dir)

        bkg_fit.save_lightcurves(lightcurves_dir)


        
