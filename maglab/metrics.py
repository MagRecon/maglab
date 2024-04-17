import numpy as np

def norm(v):
    return np.sqrt(np.sum(v**2, axis=0))

def rms(A,B):
    err = A-B
    l2 = lambda x : np.mean(x[:]**2)
    return l2(err) / l2(A)

def mean_diff_length(v1,v2):
    vdiff_norm = norm(v1-v2)
    return np.mean(vdiff_norm)


def cos_similarity(v1,v2):
    angle_diff = np.arccos(np.sum(v1*v2, axis=0))
    return np.sum(angle_diff) / np.sum(norm(v1))