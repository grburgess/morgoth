import luigi
import os
import yaml

from morgoth.utils.env import get_env_value
from morgoth.balrog_handlers import ProcessFitResults

from morgoth.utils.plot_utils import (
    create_corner_loc_plot,
    create_corner_all_plot,
    mollweide_plot, azimuthal_plot_sat_frame, interactive_3D_plot)

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")


class CreateLightcurve(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    detector = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):

        filename = f"{self.grb_name}_lightcurve_{self.report_type}_detector_{self.detector}_plot_{self.version}.png"

        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename)

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
            "b1": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b1", version=self.version),
            "b2": CreateLightcurve(grb_name=self.grb_name, report_type=self.report_type, detector="b2", version=self.version)
        }

    def output(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename))

    def run(self):
        filename = f"{self.report_type}_{self.version}_plot_all_lightcurves.txt"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, filename)

        os.system(f"touch {tmp}")


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
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        create_corner_loc_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            datapath=self.input()['post_equal_weights'].path,
            model=result['localization']['model']
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
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        create_corner_all_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            datapath=self.input()['post_equal_weights'].path,
            model=result['localization']['model']
        )

class CreateMollLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{base_dir}/{self.grb_name}/{self.report_type}/{self.version}/plots/" \
            f"{self.grb_name}_molllocation_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(filename)

    def run(self):
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        mollweide_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            trigdat_file=f"{base_dir}/{self.grb_name}/glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit",
            post_equal_weigts_file=self.input()['post_equal_weights'].path,
            used_dets=result['localization']['used_detectors'],
            model=result['localization']['model'],
            ra=result['localization']['ra'],
            dec=result['localization']['dec'],
            swift=result['general']['swift']
        )


class CreateSatellitePlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_satellite_plot_{self.report_type}_{self.version}.png"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        azimuthal_plot_sat_frame(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            trigdat_file=f"{base_dir}/{self.grb_name}/glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit",
            ra=result['localization']['ra'],
            dec=result['localization']['dec'],
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
        filename = f"{self.grb_name}_spectrum_plot_{self.report_type}_{self.version}.png"
        tmp = os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename)

        os.system(f"touch {tmp}")


class Create3DLocationPlot(luigi.Task):
    grb_name = luigi.Parameter()
    report_type = luigi.Parameter()
    version = luigi.Parameter(default="v00")

    def requires(self):
        return ProcessFitResults(grb_name=self.grb_name, report_type=self.report_type, version=self.version)

    def output(self):
        filename = f"{self.grb_name}_3dlocation_plot_{self.report_type}_{self.version}.html"
        return luigi.LocalTarget(os.path.join(base_dir, self.grb_name, self.report_type, self.version, 'plots', filename))

    def run(self):
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        interactive_3D_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            trigdat_file=f"{base_dir}/{self.grb_name}/glg_trigdat_all_bn{self.grb_name[3:]}_{self.version}.fit",
            post_equal_weigts_file=self.input()['post_equal_weights'].path,
            used_dets=result['localization']['used_detectors'],
            model=result['localization']['model'],
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
        with self.input()['result'].open() as f:
            result = yaml.safe_load(f)

        swift_gbm_plot(
            grb_name=self.grb_name,
            report_type=self.report_type,
            version=self.version,
            post_equal_weigts_file=self.input()['post_equal_weights'].path,
            model=result['localization']['model'],
            ra=result['localization']['ra'],
            dec=result['localization']['dec'],
            swift=result['general']['swift']
        )
