from morgoth.utils.file_utils import if_directory_not_existing_then_make
from morgoth.handlers.grb_handler import GRBEventHandler

import numpy as np

import time
from watchdog.observers import Observer



path = 'test_data'


if_directory_not_existing_then_make(path)


def test_basic():

    obs = Observer()

    handler = GRBEventHandler(path)


    obs.schedule(handler, path, recursive=False)

    obs.start()
    
    time.sleep(5)

    if_directory_not_existing_then_make('test_data/GRB123')

    time.sleep(5)

    if_directory_not_existing_then_make('test_data/GRB123/v00')

    time.sleep(10)



    
    obs.stop()

    obs.join()


