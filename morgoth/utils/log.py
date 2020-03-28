import logging

import coloredlogs

coloredlogs.install(
    level="INFO",
    #                    fmt="%(levelname)s:%(message)s"
)

logger = logging.getLogger("morgoth")
