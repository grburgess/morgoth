import os

import luigi
import yaml

from morgoth.balrog_handlers import ProcessFitResults
from morgoth.plots import (
    Create3DLocationPlot,
    CreateBalrogSwiftPlot,
    CreateCornerPlot,
    CreateLightcurve,
    CreateLocationPlot,
    CreateMollLocationPlot,
    CreateSatellitePlot,
    CreateSpectrumPlot
)
from morgoth.data_files import (
    CreateHealpixSysErr,
    CreateHealpix
)
from morgoth.configuration import morgoth_config
from morgoth.utils.file_utils import if_dir_containing_file_not_existing_then_make
from morgoth.utils.env import get_env_value
from morgoth.utils.upload_utils import upload_grb_report, upload_plot

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class UploadReport(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter()

    def requires(self):
        return ProcessFitResults(
            grb_name=self.grb_name, report_type=self.report_type, version=self.version
        )

    def output(self):
        filename = f"{self.report_type}_{self.version}_report.yml"
        return luigi.LocalTarget(
            os.path.join(
                base_dir, self.grb_name, self.report_type, self.version, filename
            )
        )

    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)

        report = upload_grb_report(
            grb_name=self.grb_name,
            result=result,
            wait_time=float(
                morgoth_config["upload"]["report"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["report"]["max_time"]
            ),
        )

        report_name = f"{self.report_type}_{self.version}_report.yml"
        report_path = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, report_name
        )

        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)

class UploadAllDataFiles(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "healpix": UploadHealpix(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "healpixSysErr": UploadHealpixSysErr(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_datafiles.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_datafiles.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")

class UploadHealpix(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "data_file": CreateHealpix(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector=self.detector,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_healpix.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

class UploadHealpixSysErr(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "data_file": CreateHealpixSysErr(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector=self.detector,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_healpixsyserr.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

class UploadAllPlots(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "lightcurves": UploadAllLightcurves(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "location": UploadLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "corner": UploadCornerPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "molllocation": UploadMollLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "satellite": UploadSatellitePlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "spectrum": UploadSpectrumPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "3d_location": Upload3DLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "balrogswift": UploadBalrogSwiftPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadAllLightcurves(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "n0": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n0",
                version=self.version,
            ),
            "n1": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n1",
                version=self.version,
            ),
            "n2": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n2",
                version=self.version,
            ),
            "n3": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n3",
                version=self.version,
            ),
            "n4": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n4",
                version=self.version,
            ),
            "n5": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n5",
                version=self.version,
            ),
            "n6": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n6",
                version=self.version,
            ),
            "n7": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n7",
                version=self.version,
            ),
            "n8": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n8",
                version=self.version,
            ),
            "n9": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="n9",
                version=self.version,
            ),
            "na": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="na",
                version=self.version,
            ),
            "nb": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="nb",
                version=self.version,
            ),
            "b0": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="b0",
                version=self.version,
            ),
            "b1": UploadLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector="b1",
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateLightcurve(
                grb_name=self.grb_name,
                report_type=self.report_type,
                detector=self.detector,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="lightcurve",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
            det_name=self.detector,
        )

        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_location.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="location",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_location.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadCornerPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateCornerPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_corner.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="allcorner",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_corner.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadMollLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateMollLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):
        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="molllocation",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadSatellitePlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateSatellitePlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="satellite",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadSpectrumPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateSpectrumPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="spectrum",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class Upload3DLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": Create3DLocationPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="3dlocation",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")


class UploadBalrogSwiftPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "create_report": UploadReport(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
            "plot_file": CreateBalrogSwiftPlot(
                grb_name=self.grb_name,
                report_type=self.report_type,
                version=self.version,
            ),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_balrogswift.done"
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                self.grb_name,
                self.report_type,
                self.version,
                "upload",
                filename,
            )
        )

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="balrogswift",
            version=self.version,
            wait_time=float(
                morgoth_config["upload"]["plot"]["interval"]
            ),
            max_time=float(
                morgoth_config["upload"]["plot"]["max_time"]
            ),
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_balrogswift.done"
        tmp = os.path.join(
            base_dir, self.grb_name, self.report_type, self.version, "upload", filename
        )

        if_dir_containing_file_not_existing_then_make(tmp)
        os.system(f"touch {tmp}")
