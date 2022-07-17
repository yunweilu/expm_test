import numpy as np
import scipy as sci

def get_s(A, tol):
    s = 1
    a = _exact_inf_norm(A)
    while (1):
        norm_A = a / s
        max_term_notation = np.floor(norm_A)
        max_term = 1
        for i in range(1, np.int(max_term_notation)):
            max_term = max_term * norm_A / i
            if max_term >= 10 ** 16:
                break
        if 10 ** -16 * max_term <= tol:
            break
        s = s + 1
    return s

def skew_expm(A, B=None, tol=None):
    if B == None:
        B = np.identity(A.shape[0])
    if tol == None:
        tol = 1e-16
    s = get_s(A, tol)
    F = B
    c1 = _exact_inf_norm(B)
    j = 0
    while (1):
        coeff = s * (j + 1)
        B = A.dot(B) / coeff
        c2 = _exact_inf_norm(B)
        F = F + B
        if (c1 + c2) < tol:
            m = j + 1
            break
        c1 = c2
        j = j + 1
    B = F
    for i in range(1, int(s)):
        for j in range(m):
            coeff = s * (j + 1)
            B = A.dot(B) / coeff
            F = F + B
        B = F
    return F

def _exact_inf_norm(A):
    if sci.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        return np.linalg.norm(A, np.inf)

