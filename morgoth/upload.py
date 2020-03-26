import luigi
import os
import yaml

from morgoth.utils.env import get_env_value
from morgoth.utils.upload_utils import (
    upload_grb_report,
    upload_plot,
    upload_datafile
)
from morgoth.balrog_handlers import ProcessFitResults
from morgoth.plots import (
    CreateLightcurve,
    CreateLocationPlot,
    CreateCornerPlot,
    CreateMollLocationPlot,
    CreateSatellitePlot,
    CreateSpectrumPlot,
    Create3DLocationPlot,
    CreateBalrogSwiftPlot)

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class UploadReport(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter()

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_report.yml"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        report = upload_grb_report(grb_name=self.grb_name, result=result)

        report_name = f"{self.report_type}_{self.version}_report.yml"
        report_path = os.path.join(base_dir, self.grb_name, self.report_type, self.version, report_name)

        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)


class UploadAllPlots(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
        #"lightcurves": UploadAllLightcurves(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "location": UploadLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "corner": UploadCornerPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "molllocation": UploadMollLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "satellite": UploadSatellitePlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "spectrum": UploadSpectrumPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "3d_location": Upload3DLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
        "balrogswift": UploadBalrogSwiftPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadAllLightcurves(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "n0": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n0", version=self.version),
            "n1": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n1", version=self.version),
            "n2": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n2", version=self.version),
            "n3": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n3", version=self.version),
            "n4": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n4", version=self.version),
            "n5": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n5", version=self.version),
            "n6": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n6", version=self.version),
            "n7": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n7", version=self.version),
            "n8": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n8", version=self.version),
            "n9": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n9", version=self.version),
            "na": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="na", version=self.version),
            "nb": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="nb", version=self.version),
            "b0": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b0", version=self.version),
            "b1": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b1", version=self.version),
            "b2": UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b2", version=self.version),
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector=self.detector, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='lightcurve',
            version=self.version,
            det_name=self.detector
        )

        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_location.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='location',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_location.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadCornerPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateCornerPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_corner.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='allcorner',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_corner.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadMollLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateMollLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='molllocation',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadSatellitePlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateSatellitePlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='satellite',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class UploadSpectrumPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateSpectrumPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='spectrum',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class Upload3DLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': Create3DLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='3dlocation',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")

class UploadBalrogSwiftPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'plot_file': CreateBalrogSwiftPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_balrogswift.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=self.input()['plot_file'].path,
            plot_type='balrogswift',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_balrogswift.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")