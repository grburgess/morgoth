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
    Create3DLocationPlot
)

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class UploadReport(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter()

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_report.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        with self.input()['report'].open() as f:
            report = yaml.safe_load(f)

        upload_grb_report(grb_name=self.grb_name, report=report)

        filename = f"{self.report_type}_{self.version}_upload_report.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)
        os.system(f"touch {tmp}")


class UploadAllPlots(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        yield UploadAllLightcurves(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadCornerPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadMollLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadSatellitePlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadSpectrumPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield Upload3DLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class UploadAllLightcurves(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n0", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n1", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n2", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n3", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n4", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n5", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n6", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n7", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n8", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n9", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="na", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="nb", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b0", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b1", version=self.version)
        yield UploadLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b2", version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_all_lightcurves.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class UploadLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'create_report': CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector=self.detector, version=self.version),
            'plot_file': UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='lightcurve',
            version=self.version,
            det_name=self.detector
        )

        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='location',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_location.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='allcorner',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_corner.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='molllocation',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='satellite',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='spectrum',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

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
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_file = self.input()['plot_file']

        upload_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            plot_file=plot_file,
            plot_type='3dlocation',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")
