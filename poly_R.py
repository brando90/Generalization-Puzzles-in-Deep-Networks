import numpy as np
import scipy as sc

def poly(x, degree):
    xbar = np.mean(x)
    x = x - xbar

    # R: outer(x, 0L:degree, "^")
    X = x[:, None] ** np.arange(0, degree+1)

    #R: qr(X)$qr
    q, r = np.linalg.qr(X)

    #R: r * (row(r) == col(r))
    z = np.diag((np.diagonal(r)))

    # R: Z = qr.qy(QR, z)
    Zq, Zr = np.linalg.qr(q)
    Z = np.matmul(Zq, z)

    # R: colSums(Z^2)
    norm1 = (Z**2).sum(0)

    #R: (colSums(x * Z^2)/norm2 + xbar)[1L:degree]
    alpha = ((x[:, None] * (Z**2)).sum(0) / norm1 +xbar)[0:degree]

    # R: c(1, norm2)
    norm2 = np.append(1, norm1)

    # R: Z/rep(sqrt(norm1), each = length(x))
    Z = Z / np.reshape(np.repeat(norm1**(1/2.0), repeats = x.size), (-1, x.size), order='F')

    #R: Z[, -1]
    Z = np.delete(Z, 0, axis=1)
    return [Z, alpha, norm2];


x = np.arange(10) + 1
degree = 9
p = poly(x, degree)
#poly(np.arange(10)+1, 9)
print(p[0])
