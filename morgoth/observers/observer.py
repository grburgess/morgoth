import time
from watchdog.observers import Observer

import coloredlogs, logging
import morgoth.utils.log

logger = logging.getLogger('morgoth.observer')


class ObserverBuilder(object):
    def __init__(self, handler, path, interval=1, max_time=60 * 60):


        logger.debug('Building a new observer')
        
        self._interval = interval
        self._max_time = max_time

        self._observer = Observer()

        self._handler = handler

        self._handler.set_observer(self)
        
        self._observer.schedule(self._handler, path, recursive=False)

        self._observer.start()

        self._flag = True

        self._total_time = 0

        try:

            while self._flag:

                time.sleep(self._interval)

                self._total_time += self._interval

                if self._total_time > self._max_time:

                    self._flag = False
                    self._observer.stop()

                if self._handler.is_complete:

                    self._flag = False
                    self._observer.stop()

        except KeyboardInterrupt:

            self._observer.stop()

        self._observer.join()
