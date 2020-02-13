import os

from morgoth.handlers.smart_handler import SmartHandler

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger("morgoth.folderevent")


class DirectoryHandler(SmartHandler):
    def __init__(self, base_directory):

        self._base_directory = base_directory

        super(DirectoryHandler, self).__init__()

    @property
    def base_directory(self):
        return self._base_directory

    def on_any_event(self, event):

        if event.is_directory and (event.event_type == "created"):

            # get the directory name
            _, self._dir_name = os.path.split(event._src_path)

            self._process()

    def _process(self):

        raise NotImplementedError()
