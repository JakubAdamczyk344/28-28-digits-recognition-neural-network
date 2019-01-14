"""
mnistLoader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``loadData``
and ``loadDataWrapper``.  In practice, ``loadDataWrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def loadData():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``trainingData`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``trainingData`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validationData`` and ``testData`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``trainingData`` a little.
    That's done in the wrapper function ``loadDataWrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = cPickle.load(f, encoding="latin1")
    f.close()
    return (trainingData, validationData, testData)

def loadDataWrapper():
    """Return a tuple containing ``(trainingData, validationData,
    testData)``. Based on ``loadData``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``trainingData`` is an iterator containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validationData`` and ``testData`` are iterators containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    trData, vaData, teData = loadData()
    trainingInputs = [np.reshape(x, (784, 1)) for x in trData[0]]
    trainingResults = [vectorizedResult(y) for y in trData[1]]
    trainingData = zip(trainingInputs, trainingResults)
    validationInputs = [np.reshape(x, (784, 1)) for x in vaData[0]]
    validationData = zip(validationInputs, vaData[1])
    testInputs = [np.reshape(x, (784, 1)) for x in teData[0]]
    testData = zip(testInputs, teData[1])
    return (trainingData, validationData, testData)

def vectorizedResult(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e