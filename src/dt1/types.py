import numpy

FeatureVector = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool]]
"""0-indexed feature vector for a single sample"""

FeatureMatrix = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]]
"""0-indexed feature matrix for multiple samples"""

LabelVector = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool]]
"""0-indexed label vector for multiple samples"""

TreeNodeInfo = numpy.ndarray[tuple[int], numpy.dtype[numpy.int32]]
"""1-indexed tree node info"""
