import os

from morgoth.handlers.smart_handler import SmartHandler

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger("morgoth.folderevent")


class DirectoryHandler(SmartHandler):
    def __init__(self, base_directory, *args, **kwargs):

        self._base_directory = base_directory

        super(DirectoryHandler, self).__init__(*args, **kwargs)

        logger.debug(f'I am new directory handler at {base_directory}')
        
    @property
    def base_directory(self):
        return self._base_directory

    def on_any_event(self, event):

        logger.debug('I GOT AN EVENT')
        
        if event.is_directory and (event.event_type == "created"):

            # get the directory name
            _, self._dir_name = os.path.split(event._src_path)

            logger.debug(f' I found a new folder {self._dir_name}')
            
            self._process()

    def _process(self):

        raise NotImplementedError()
