import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        ls = np.shape(y)[0]
        X = X @ w * (-y)
        X = np.logaddexp(0, X)
        return np.sum(X)/ls + self.l2_coef/2 * np.dot(w, w)
        """
        log(exp(x1) + exp(x2)) = numpy.logaddexp(x1, x2)
        scipy.special.logsumexp(a, axis=None) =  the log of the sum of exponentials of input elements.
        scipy.special.expit(x) = expit(x) = 1/(1+exp(-x))
        Вычислить значение функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """

    def grad(self, X, y, w):
        ls = np.shape(y)[0]
        Z = X @ w * y
        Z = scipy.special.expit(Z) * np.exp(-Z) * (-y)
        Z = np.sum(X * Z[:, np.newaxis], axis=0)/ls
        return Z + self.l2_coef * w
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w - одномерный numpy array
        """
