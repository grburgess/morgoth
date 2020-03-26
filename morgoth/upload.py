import luigi
import os
import yaml

from morgoth.utils.upload_utils import (
    upload_grb_report,
    upload_plot,
    upload_datafile
)
from morgoth.reports import CreateReportTTE, CreateReportTrigdat
from morgoth.plots import (
    CreateLightcurve,
    CreateLocationPlot,
    CreateCornerPlot,
    CreateMollLocationPlot,
    CreateSatellitePlot,
    CreateSpectrumPlot,
    Create3DLocationPlot
)

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")

class UploadAllPages(luigi.WrapperTask):
    grb_name = luigi.Parameter()

    def requires(self):
        yield UploadPage(grb_name=self.grb_name, report_type="tte")
        yield UploadPage(grb_name=self.grb_name, report_type="trigdat", version="v00")
        yield UploadPage(grb_name=self.grb_name, report_type="trigdat", version="v01")
        yield UploadPage(grb_name=self.grb_name, report_type="trigdat", version="v02")


class UploadPage(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        yield UploadAllPlots(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_page.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_upload_page.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")


class UploadReport(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        if self.report_type.lower() == "tte":
            return CreateReportTTE(grb_name=self.grb_name)

        elif self.report_type.lower() == "trigdat":
            return CreateReportTrigdat(grb_name=self.grb_name, version=self.version)

        else:
            return None

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_report.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        report_name = f"{self.report_type}_{self.version}_report.yml"
        report_file = os.path.join(base_dir, self.grb_name, report_name)

        with open(report_file) as f:
            report = yaml.load(f)

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
        yield CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector=self.detector, version=self.version)
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_{self.detector}_upload_plot_lightcurve.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield CreateLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_location.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_location_{self.report_type}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield CreateCornerPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_corner.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_allcorner_{self.report_type}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield CreateMollLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_molllocation.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_molllocation_{self.report_type}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield CreateSatellitePlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_satellite.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_satellite_{self.report_type}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield CreateSpectrumPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_spectrum.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        plot_name = f"{self.grb_name}_spectrum_{self.report_type}_plot_{self.version}.png"
        plot_file = os.path.join(base_dir, self.grb_name, plot_name)

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
        yield UploadReport(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        yield Create3DLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, filename))

    def run(self):
        data_file_name = f"{self.grb_name}_3dlocation_{self.report_type}_plot_{self.version}.png"
        data_file = os.path.join(base_dir, self.grb_name, data_file_name)

        upload_datafile(
            grb_name=self.grb_name,
            report_type=self.report_type,
            data_file=data_file,
            file_type='3dlocation',
            version=self.version
        )

        filename = f"{self.report_type}_{self.version}_upload_plot_3dlocation.txt"
        tmp = os.path.join(base_dir, self.grb_name, filename)

        os.system(f"touch {tmp}")

