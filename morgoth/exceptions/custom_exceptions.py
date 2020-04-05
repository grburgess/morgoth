class GRBNotFound(RuntimeError):
    pass


class DBConflict(RuntimeError):
    pass


class EmptyFileError(RuntimeError):
    pass


class ImproperlyConfigured(RuntimeError):
    pass


class UnkownReportType(RuntimeError):
    pass


class UnauthorizedRequest(RuntimeError):
    pass


class UnexpectedStatusCode(RuntimeError):
    pass


class UploadFailed(RuntimeError):
    pass
