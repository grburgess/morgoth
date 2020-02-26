import gcn
import time
import os

import luigi
from morgoth.reports import CreateAllPages
from morgoth.trigger import parse_trigger_file_and_write

@gcn.include_notice_types(
    gcn.notice_types.FERMI_GBM_FLT_POS,  # Fermi GBM localization (flight)
)
def handler(payload, root):

    grb = parse_trigger_file_and_write(root)

    luigi.build(
        [CreateAllPages(grb_name=grb)],
        local_scheduler=False,
        scheduler_host="localhost",
        workers=14,
    )


