# import os
# from morgoth.utils.file_utils import if_directory_not_existing_then_make

# # from morgoth.handlers.grb_handler import GRBEventHandler

# from morgoth.handler import GRBHandler


# import numpy as np

# import time
# from watchdog.observers import Observer


# path = "test_data"


# if_directory_not_existing_then_make(path)


# def test_basic():

#     obs = Observer()

#     handler = GRBHandler(path)

#     obs.schedule(handler, path, recursive=True)

#     obs.start()

#     time.sleep(2)

#     if_directory_not_existing_then_make("test_data/GRB123")

#     time.sleep(2)

#     if_directory_not_existing_then_make("test_data/GRB123/v00")

#     time.sleep(2)

#     os.system('touch test_data/GRB123/v00/glg_trigdat_fuck_you.fit')


#     time.sleep(2)
    
#     obs.stop()

#     obs.join()
