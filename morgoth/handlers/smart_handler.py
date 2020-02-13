import functools
from watchdog.events import FileSystemEventHandler


def you_complete_me(method):
    @functools.wraps(method)
    def wrapper(instance, *arg, **kwargs):

        method(instance, *arg, **kwargs)

        instance._is_complete = True

    return wrapper


class SmartHandler(FileSystemEventHandler):
    def __init__(self, *args, **kwargs):

        self._is_complete = False

        super(SmartHandler, self).__init__(*args, **kwargs)

    @property
    def is_complete(self):

        return self._is_complete
