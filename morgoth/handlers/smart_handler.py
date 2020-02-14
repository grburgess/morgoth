import functools
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

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
