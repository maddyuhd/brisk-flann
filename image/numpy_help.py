import numpy as np


def hamming2(s1, s2):
    r = (1 << np.arange(8))[:, None]
    return np.count_nonzero((np.bitwise_xor(s1, s2) & r) != 0)


def featureCleanup(data):
    data = eval("[" + data[1:-2].replace("\;", "],[") + "]]")
    data = np.asarray(data)
    data = data.astype("uint8")
    return data


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]
