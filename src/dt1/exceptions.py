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


class UpperBoundTooStrictError(DT1Exception):
    """Raised when the provided upper bound for decision tree size is too strict to build a valid tree."""

    def __init__(self, message: str):
        super().__init__(message)
