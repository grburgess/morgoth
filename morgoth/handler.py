import os
import functools
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from morgoth.proccesor import ProcessTrigdat, ProcessTTE

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger("morgoth.grb_handler")


def you_complete_me(method):
    @functools.wraps(method)
    def wrapper(instance, *arg, **kwargs):

        method(instance, *arg, **kwargs)

        instance._is_complete = True

    return wrapper


class SmartHandler(FileSystemEventHandler):
    def __init__(self, *args, **kwargs):

        self._is_complete = False
        self._observer = None
        super(SmartHandler, self).__init__(*args, **kwargs)

    @property
    def is_complete(self):

        return self._is_complete

    def set_observer(self,  observer):

        assert isinstance(observer, Observer)
        
        self._observer = observer




class GRBHandler(SmartHandler):
    def __init__(self, base_directory, *args, **kwargs):

        self._base_directory = base_directory

        self._active_grbs = {}

        super(GRBHandler, self).__init__(*args, **kwargs)

        logger.debug(f'I am new directory handler at {base_directory}')
        
    @property
    def base_directory(self):
        return self._base_directory

    def on_any_event(self, event):

        logger.debug('I GOT AN EVENT')
        
        if event.is_directory and (event.event_type == "created"):

            # get the directory name
            head, dir_name = os.path.split(event._src_path)


            if 'GRB' in dir_name:


            
                logger.info(f'There is a new {dir_name}')
            
                self._active_grbs[dir_name] = True

            elif 'v0' in dir_name:

                _, grb = os.path.split(head)

                logger.info(f'Well {grb} now has some {dir_name} data')

                
            else:

                logger.debug(f'Found directory {event._src_path}')
                
                pass

        elif (event.event_type == "created"):

            head, file_name = os.path.split(event._src_path)

            logging.debug(f'found a file named {file_name}')
            
            if 'trigdat' in  file_name:

                logger.info(f'{file_name} has arrived.')

        else:

            pass
                
                
        

    def _process(self):

        raise NotImplementedError()
