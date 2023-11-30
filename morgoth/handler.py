import shlex
import subprocess
import os
import gcn

from morgoth import morgoth_config
from morgoth.trigger import parse_trigger_file_and_write

n_workers = int(morgoth_config["luigi"]["n_workers"])


@gcn.include_notice_types(
    gcn.notice_types.FERMI_GBM_FLT_POS,  # Fermi GBM localization (flight)
)
def handler(payload, root):
    """
    The pygcn handler

    :param payload:
    :param root:
    :returns:
    :rtype:

    """

    # parse the trigger XML file
    # and write to yaml

    grb = parse_trigger_file_and_write(root, payload)

    # form the luigi command

    cmd = form_morgoth_cmd_string(grb)

    # launch luigi
    os.system(" ".join(cmd))
    # subprocess.Popen(cmd)


def form_morgoth_cmd_string(grb):
    """
    makes the command string for luigi

    :param grb:
    :returns:
    :rtype:

    """

    base_cmd = "luigi --module morgoth "

    cmd = f"{base_cmd} CreateAllPages --grb-name {grb} "

    cmd += f"--workers {n_workers} --scheduler-host localhost --log-level INFO &"

    cmd = shlex.split(cmd)

    return cmd
