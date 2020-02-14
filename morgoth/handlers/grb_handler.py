import os
from morgoth.handlers.directory_handler import DirectoryHandler
from morgoth.handlers.version_watcher import VersionEventHandler
from morgoth.observers.observer import ObserverBuilder



import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger('morgoth.grbevent')


class GRBEventHandler(DirectoryHandler):

    def _process(self):


        if 'GRB' in self._dir_name:

            # now we launch the directory
            # watch for versions
            logger.info(f'Detected data from {self._dir_name}!')

            path =  os.path.join(self._base_directory, self._dir_name)

            logger.debug(f"launching a version watcher a {path}")
            
            version_handler = VersionEventHandler(base_directory=path)
            version_observer = ObserverBuilder(version_handler,
                                               path=path,
                                               interval = 1,
                                               max_time = 60*60*24, # one day
                                               

            )
        
