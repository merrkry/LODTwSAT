import numpy

from dt1.builder import BuildResult, TimeoutBehavior, build_dt1_classifier
from dt1.exceptions import (
    InvalidTestSetError,
    InvalidTrainingSetError,
)
from dt1.tree import DecisionTree
from dt1.types import FeatureMatrix, FeatureVector, LabelVector


def _check_consistent(features: FeatureMatrix, labels: LabelVector) -> None:
    """
    Verify the training set is consistent: no duplicate features with different labels.
    Raises InvalidTrainingSetError if inconsistent.

    Uses vectorized numpy operations for efficiency on larger datasets.
    """
    # Find unique rows and the inverse mapping (which original row maps to which unique)
    unique_features, inverse = numpy.unique(features, axis=0, return_inverse=True)

    # For each unique feature, check if all corresponding labels are the same
    for i in range(len(unique_features)):
        mask = inverse == i  # Boolean mask for rows matching the i-th unique feature
        label_group = labels[mask]  # Corresponding labels where mask is True
        unique_label_vals = numpy.unique(label_group)
        if len(unique_label_vals) > 1:
            feature_tuple = tuple(unique_features[i])
            raise InvalidTrainingSetError(
                f"Inconsistent training set: feature {feature_tuple} has conflicting labels."
            )


class DT1Classifier:
    _n_samples: int
    _n_features: int
    _decision_tree: DecisionTree
    _build_result: BuildResult

    def __init__(
        self,
        features: FeatureMatrix,
        labels: LabelVector,
        max_size: int | None = None,
        *,
        timeout: float | None = None,
        solver: str = "glucose3",
        verbose: bool = False,
        timeout_behavior: TimeoutBehavior = TimeoutBehavior.ERROR,
    ) -> None:
        """
        Initialize and train a DT1 classifier.

        Args:
            features: training features (binary matrix)
            labels: training labels (binary vector)
            max_size: upper bound of decision tree size.
                If None, scikit's decision tree builder will be called for upper bound approximation.
            timeout: total timeout in seconds (default: 60)
            solver: name of SAT solver to use (default: "glucose3")
            verbose: if True, print progress information
            timeout_behavior: behavior when timeout is reached (default: ERROR)

        Raises:
            InvalidTrainingSetError: if features/labels mismatch or inconsistent dataset
            UpperBoundTooStrictError: if user-provided max_size is too strict or
                timeout is reached with no valid tree (when timeout_behavior is ERROR)
        """
        if features.shape[0] != labels.shape[0]:
            raise InvalidTrainingSetError(
                "Inconsistent number of samples between features and labels."
            )

        _check_consistent(features, labels)

        self._n_samples = features.shape[0]
        self._n_features = features.shape[1]

        result = build_dt1_classifier(
            features,
            labels,
            max_size,
            timeout=timeout,
            solver=solver,
            verbose=verbose,
            timeout_behavior=timeout_behavior,
        )
        self._decision_tree = result.tree
        self._build_result = result  # Store for access to timing info

    @property
    def build_result(self) -> BuildResult:
        """Return the build result containing timing information."""
        return self._build_result

    def predict(self, features: FeatureMatrix) -> LabelVector:
        if features.shape[1] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test set does not match training set."
            )

        return numpy.array([self.predict_single(row) for row in features])

    def predict_single(self, features: FeatureVector) -> bool:
        if features.shape[0] != self._n_features:
            raise InvalidTestSetError(
                "Number of features in test sample does not match training set."
            )

        return self._decision_tree.predict(features)
