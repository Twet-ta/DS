import numpy as np
import scipy
from scipy.special import expit
import timeit
from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0, tolerance=1e-5, max_iter=1000, **kwargs):
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """

    def fit(self, X, y, w_0=None, trace=False):
        if w_0 is None:
            w_0 = 0
        w = w_0
        BL = BinaryLogistic(**self.kwargs)
        ls = np.shape(y)[0]
        if trace:
            history = {}
            history['time'] = []
            history['func'] = []
            a = timeit.default_timer()
            f_st = BL.func(X, y, w)
            history['func'].append(round(f_st, 12))
            history['time'].append(timeit.default_timer()-a)
            for i in range(self.max_iter):
                a = timeit.default_timer()
                n = self.step_alpha/(i + 1)**self.step_beta
                w = w - n*BL.grad(X, y, w)
                f_fin = BL.func(X, y, w)
                history['func'].append(round(f_fin, 12))
                history['time'].append(timeit.default_timer()-a)
                if abs(f_fin - f_st) < self.tolerance:
                    break
                f_st = f_fin
            return history
        else:
            f_st = BL.func(X, y, w)
            for i in range(self.max_iter):
                n = self.step_alpha/(i + 1)**self.step_beta
                w = w - n*BL.grad(X, y, w)
                f_fin = BL.func(X, y, w)
                if abs(f_fin - f_st) < self.tolerance:
                    break
                f_st = f_fin
        self.w = w
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)

        a = timeit.default_timer()
        get_max_before_zero_v(A)
        t_v.append(timeit.default_timer()-a)
        """

    def predict(self, X):
        def sign(x):
            if x > 0:
                return 1
            else:
                return 0
        return sign(np.dot(self.w, X))
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """

    def predict_proba(self, X):
        Z = np.zeros((np.shape(X)[0], 2))
        Z[1] = scipy.special.expit(X @ w)
        Z[0] = 1 - Z[1]
        return Z
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """

    def get_objective(self, X, y):
        BL = BinaryLogistic(**self.kwargs)
        return BL.func(X, y, self.w)
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """

    def get_gradient(self, X, y):
        BL = BinaryLogistic(**self.kwargs)
        return BL.grad(X, y, self.w)
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """

    def get_weights(self):
        return self.w
        """
        Получение значения весов функционала
        """


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
