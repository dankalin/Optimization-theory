import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):  # оценка функционала
        # TODO: Implement
        logadd = np.logaddexp(0, -1 * self.b * self.matvec_Ax(x))
        res = np.linalg.norm(logadd, 1) / self.b.size + \
              np.linalg.norm(x, 2) ** 2 * self.regcoef / 2
        return res

    def grad(self, x):
        # TODO: Implement
        return self.regcoef * x - self.matvec_ATx(self.b * (expit(-self.b * self.matvec_Ax(x)))) / self.b.size

    def hess(self, x):
        # TODO: Implement
        tmp = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(tmp * (1 - tmp)) / self.b.size + self.regcoef * np.identity(x.size)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        z = self.matvec_Ax(x)
        z_d = self.matvec_Ax(d)
        return np.mean(np.logaddexp(0, -self.b * (z + alpha * z_d))) + \
            self.regcoef / 2 * np.dot(x + alpha * d, x + alpha * d)

    def grad_directional(self, x, d, alpha):
        m = self.b.shape[0]
        z = self.b * self.matvec_Ax(x)
        z_d = self.b * self.matvec_Ax(d)
        s = scipy.special.expit(z + alpha * z_d)
        return -1 / m * np.dot(self.b * s, z_d) + self.regcoef * np.dot(x + alpha * d, d)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    if scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)
        # redefine if matrix was not CSR
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        matmat_ATsA = lambda x: matvec_ATx(matvec_ATx(scipy.sparse.diags(x)).T)
    else:
        matmat_ATsA = lambda x: np.dot(matvec_ATx(np.diag(x)), A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-9):
    x = np.asarray(x)
    fx = func(x)
    res, e = np.zeros(x.size), np.identity(x.size)

    for i in range(x.size):
        res[i] = func(x + eps * e[i]) - fx

    return res / eps


def hess_finite_diff(func, x, eps=1e-5):
    x = np.asarray(x)
    fx = func(x)
    e = np.identity(x.size)
    res = np.zeros((x.size, x.size))
    tmp = np.zeros_like(x)

    for i in range(x.size):
        tmp[i] = func(x + e[i] * eps)

    for i in range(x.size):
        for j in range(x.size):
            res[i][j] = func(x + eps * e[i] + eps * e[j]) \
                        - tmp[i] \
                        - tmp[j] + fx

    return res / eps ** 2
