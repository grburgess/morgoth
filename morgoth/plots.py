import os

import luigi
import yaml

from morgoth.balrog_handlers import ProcessFitResults
from morgoth.bkg_fit_handler import GatherTrigdatBackgroundFit
from morgoth.downloaders import DownloadTrigdat
from morgoth.exceptions.custom_exceptions import UnkownReportType
from morgoth.utils.env import get_env_value
from morgoth.utils.plot_utils import (
    azimuthal_plot_sat_frame,
    create_corner_all_plot,
    create_corner_loc_plot,
    interactive_3D_plot,
    mollweide_plot,
    swift_gbm_plot
)

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class CreateAllPlots(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'lightcurves': CreateAllLightcurves(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'location': CreateLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'corner': CreateCornerPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'molllocation': CreateMollLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'satellite': CreateSatellitePlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'spectrum': CreateSpectrumPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            '3dlocation': Create3DLocationPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'balrogswift': CreateBalrogSwiftPlot(grb_name=self.grb_name, report_type=self.report_type, version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_plot_all.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_plot_all.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class CreateAllLightcurves(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            "n0": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n0", version=self.version),
            "n1": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n1", version=self.version),
            "n2": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n2", version=self.version),
            "n3": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n3", version=self.version),
            "n4": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n4", version=self.version),
            "n5": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n5", version=self.version),
            "n6": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n6", version=self.version),
            "n7": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n7", version=self.version),
            "n8": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n8", version=self.version),
            "n9": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="n9", version=self.version),
            "na": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="na", version=self.version),
            "nb": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="nb", version=self.version),
            "b0": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b0", version=self.version),
            "b1": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b1", version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


class CreateLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        base_job = os.path.join(base_dir, self.grb_name, self.report_type, self.version)
        filename = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_job, 'plots', 'lightcurves', filename))

    def run(self):
        # The lightcurve is created in the background fit Task, this task will check if the creation was successful
        pass


class CreateLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_location_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['result_file'].open() as f:
            result = yaml.safe_load(f)

        create_corner_loc_plot(
            post_equal_weights_file=self.input()['post_equal_weights'].path,
            model=result['fit_result']['model'],
            save_path=self.output().path
        )


class CreateCornerPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_allcorner_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['result_file'].open() as f:
            result = yaml.safe_load(f)

        create_corner_all_plot(
            post_equal_weights_file=self.input()['post_equal_weights'].path,
            model=result['fit_result']['model'],
            save_path=self.output().path
        )


class CreateMollLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'fit_result': ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'trigdat_version': GatherTrigdatBackgroundFit(grb_name=self.grb_name),
        }

    def output(self):
        filename = f"{base_dir}/{self.grb_name}/{self.report_type}/{self.version}/plots/" \
            f"{self.grb_name}_molllocation_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(filename)

    def run(self):
        with self.input()['fit_result']['result_file'].open() as f:
            result = yaml.safe_load(f)

        if self.report_type.lower() == 'tte':
            with self.input()['trigdat_version'].open() as f:
                trigdat_version = yaml.safe_load(f)['trigdat_version']

            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=trigdat_version).output()

        elif self.report_type.lower() == 'trigdat':
            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=self.version).output()

        else:
            raise UnkownReportType(f"The report_type '{self.report_type}' is not valid!")

        mollweide_plot(
            grb_name=self.grb_name,
            trigdat_file=trigdat_file.path,
            post_equal_weights_file=self.input()['fit_result']['post_equal_weights'].path,
            used_dets=result['time_selection']['used_detectors'],
            model=result['fit_result']['model'],
            ra=result['fit_result']['ra'],
            dec=result['fit_result']['dec'],
            swift=result['general']['swift'],
            save_path=self.output().path
        )


class CreateSatellitePlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'fit_result': ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'trigdat_version': GatherTrigdatBackgroundFit(grb_name=self.grb_name),
        }

    def output(self):
        filename = f"{self.grb_name}_satellite_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['fit_result']['result_file'].open() as f:
            result = yaml.safe_load(f)

        if self.report_type.lower() == 'tte':
            with self.input()['trigdat_version'].open() as f:
                trigdat_version = yaml.safe_load(f)['trigdat_version']

            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=trigdat_version).output()

        elif self.report_type.lower() == 'trigdat':
            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=self.version).output()

        else:
            raise UnkownReportType(f"The report_type '{self.report_type}' is not valid!")

        azimuthal_plot_sat_frame(
            grb_name=self.grb_name,
            trigdat_file=trigdat_file.path,
            ra=result['fit_result']['ra'],
            dec=result['fit_result']['dec'],
            save_path=self.output().path
        )


class CreateSpectrumPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_spectrum_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        # The spectrum plot is created in the balrog fit Task, this task will check if the creation was successful
        pass


class Create3DLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return {
            'fit_result': ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version),
            'trigdat_version': GatherTrigdatBackgroundFit(grb_name=self.grb_name),
        }

    def output(self):
        filename = f"{self.grb_name}_3dlocation_plot_{self.report_type}_{self.version}.html"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['fit_result']['result_file'].open() as f:
            result = yaml.safe_load(f)

        if self.report_type.lower() == 'tte':
            with self.input()['trigdat_version'].open() as f:
                trigdat_version = yaml.safe_load(f)['trigdat_version']

            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=trigdat_version).output()

        elif self.report_type.lower() == 'trigdat':
            trigdat_file = DownloadTrigdat(grb_name=self.grb_name, version=self.version).output()

        else:
            raise UnkownReportType(f"The report_type '{self.report_type}' is not valid!")

        interactive_3D_plot(
            trigdat_file=trigdat_file.path,
            post_equal_weights_file=self.input()['fit_result']['post_equal_weights'].path,
            used_dets=result['time_selection']['used_detectors'],
            model=result['fit_result']['model'],
            save_path=self.output().path
        )


class CreateBalrogSwiftPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_balrogswift_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['result_file'].open() as f:
            result = yaml.safe_load(f)

        swift_gbm_plot(
            grb_name=self.grb_name,
            post_equal_weights_file=self.input()['post_equal_weights'].path,
            model=result['fit_result']['model'],
            ra=result['fit_result']['ra'],
            dec=result['fit_result']['dec'],
            swift=result['general']['swift'],
            save_path=self.output().path
        )
