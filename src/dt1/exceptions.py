class DT1Exception(Exception):
    """Libray-specific exceptions in DT1."""


class InvalidTrainingSetError(DT1Exception):
    """Raised when the training set provided is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidTestSetError(DT1Exception):
    """Raised when the test set provided is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
