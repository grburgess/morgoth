from configya import YAMLConfig


structure = {}

structure["pygcn"] = dict(port=8099)
structure["luigi"] = dict(n_workers=4)
structure["multinest"] = dict(n_cores=4, path_to_python="python")
structure["download"] = dict(
    trigdat=dict(
        v00=dict(interval=5, max_time=1800),
        v01=dict(interval=5, max_time=1800),
        v02=dict(interval=5, max_time=1800),
    ),
    tte=dict(
        v00=dict(interval=5, max_time=7200),
        v01=dict(interval=5, max_time=7200),
        v02=dict(interval=5, max_time=7200),
    ),
    cspec=dict(
        v00=dict(interval=5, max_time=7200),
        v01=dict(interval=5, max_time=7200),
        v02=dict(interval=5, max_time=7200),
    ),
)
structure["upload"] = dict(
    report=dict(
        interval=2, max_time=1800
    ),
    plot=dict(
        interval=5, max_time=1800
    ),
    datafile=dict(
        interval=5, max_time=1800
    ),
)


class MorgothConfig(YAMLConfig):
    def __init__(self):

        super(MorgothConfig, self).__init__(
            structure=structure,
            config_path="~/.morgoth",
            config_name="morgoth_config.yml",
        )


morgoth_config = MorgothConfig()
