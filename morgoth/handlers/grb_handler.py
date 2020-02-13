import os
from morgoth.handlers.directory_handler import DirectoryHandler


from watchdog.observer import Observer

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger('morgoth.grbevent')


class GRBEventHandler(DirectoryHandler):

    def _process(self):


        if 'GRB' in self._dir_name:

            # now we launch the directory
            # watch for versions
    
        
