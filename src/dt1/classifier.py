import numpy

from dt1.exceptions import InvalidTestSetError, InvalidTrainingSetError

FeatureVector = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool]]
"""0-indexed feature vector for a single sample"""

FeatureMatrix = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]]
"""0-indexed feature matrix for multiple samples"""

LabelVector = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool]]
"""0-indexed label vector for multiple samples"""


class DT1Classifier:
    _n_samples: int
    _n_features: int

    def __init__(
        self, features: FeatureMatrix, labels: LabelVector, max_size: int
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise InvalidTrainingSetError(
                "Inconsistent number of samples between features and labels."
            )

        self._n_samples = features.shape[0]
        self._n_features = features.shape[1]

        # TODO: actually train decision tree

    def predict(self, features: FeatureMatrix) -> LabelVector:
        if features.shape[1] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test set does not match training set."
            )

        return numpy.array([self.predict_single(row) for row in features])

    def predict_single(self, features: FeatureVector) -> numpy.bool:
        if features.shape[0] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test sample does not match training set."
            )

        # TODO: actually call decision tree
        return numpy.bool(False)
