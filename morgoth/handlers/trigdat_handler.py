import os

from watchdog.events import FileSystemEventHandler

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger('morgoth.trigdat_event')


class TrigDatHandler(FileSystemEventHandler):

    def __init__(self, base_directory):

        self._base_directory = base_directory

    @property
    def base_directory(self):
        return self._base_directory

    def on_any_event(self, event):

        if (not event.is_directory) and (event.event_type == 'created'):

            # get the file name

            
           _, self._file_name = os.path.split(event._src_path)

           if 'trigdat' in self._file_name:
           
               self._process()

    def _process(self):

        print('Hey! {self._file_name} has arrived! do some shit')
