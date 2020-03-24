
class GRBNotFound(RuntimeError):
    pass


class DBConflict(RuntimeError):
    pass


class EmptyFileError(RuntimeError):
    pass


class ImproperlyConfigured(RuntimeError):
    pass