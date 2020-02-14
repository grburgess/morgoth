import os
import numpy as np
from morgoth.observers.observer import ObserverBuilder
from morgoth.handlers.trigdat_handler import TrigDatHandler
from morgoth.handlers.directory_handler import DirectoryHandler

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger('morgoth.version_event')


class VersionEventHandler(DirectoryHandler):

    def __init__(self, base_directory, *args, **kwargs):



        self._versions = {}

        for i in range(3):
            self._versions[f'v0{i}'] = False
        

        super(VersionEventHandler, self).__init__(base_directory, *args, **kwargs)


    
    def _process(self):


        if 'v0' in self._dir_name:

            # start the trigdat searchers
            logger.info(f'Detected version {self._dir_name}!')
            trigdat_handler = TrigDatHandler()
            trigdat_observer = ObserverBuilder(trigdat_observer, path=path)

            self._versions[self._dir_name] = True

            
            test = []
            for k, v in self._versions.items():

                test.append(v)

            if np.all(test):

                self._is_complete = True

            
            

            # now we launch the directory
            # watch for versions
    
        
