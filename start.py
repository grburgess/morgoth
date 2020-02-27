from morgoth import morgoth_config

n_workers = int(morgoth_config['n_workers'])


def form_morgoth_cmd_string(*grbs):

    cmd = f"luigi --module morgoth "

    for grb in grbs:

        cmd += f"CreateAllPages --grb={grb} "

    cmd += f"--workers={n_workers} &"

    return cmd
