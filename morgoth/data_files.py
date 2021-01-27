import os

import luigi
import yaml

from morgoth.exceptions.custom_exceptions import UnkownReportType
from morgoth.utils.env import get_env_value
from morgoth.balrog_handlers import ProcessFitResults

from morgoth.utils.healpix import healpix_no_sys, healpix_with_sys
from morgoth.utils.file_utils import if_dir_containing_file_not_existing_then_make

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")

class CreateHealpix(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")
    def requires(self):
        return ProcessFitResults(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version
        )

    def output(self):
        filename = (
            f"{self.grb_name}_healpix_{self.report_type}_{self.version}.fits"
        )
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "datafiles",
                filename,
            )
        )
    
    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)
        if_dir_containing_file_not_existing_then_make(self.output().path)
        healpix_no_sys(
            nside=512,
            result_path=os.path.join(base_dir, self.grb_name,
                                     self.report_type, self.version,
                                     f"{self.report_type}_{self.version}_loc_results.fits"),
            save_path=self.output().path,
        )

class CreateHealpixSysErr(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(
            grb_name=self.grb_name, report_type=self.report_type, version=self.version
        )

    def output(self):
        filename = (
            f"{self.grb_name}_healpixSysErr_{self.report_type}_{self.version}.fits"
        )
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "datafiles",
                filename,
            )
        )
    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)

        if_dir_containing_file_not_existing_then_make(self.output().path)
        
        healpix_with_sys(
            nside=512,
            n_samples_point=100,
            sat_phi=result["fit_result"]["sat_phi"],
            result_path=os.path.join(base_dir, self.grb_name,
                                     self.report_type, self.version,
                                     f"{self.report_type}_{self.version}_loc_results.fits"),
            save_path=self.output().path,
        )
