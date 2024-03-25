import numpy as np
from numpy.linalg import LinAlgError
import scipy
import time
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        def armijo():
            a = previous_alpha or self.alpha_0
            while oracle.func_directional(x_k, d_k, a) \
                    > oracle.func_directional(x_k, d_k, 0) \
                    + self.c1 * a * oracle.grad_directional(x_k, d_k, 0):
                a /= 2
            return a

        if self._method == 'Wolfe':
            a = previous_alpha or self.alpha_0
            from scipy.optimize.linesearch import scalar_search_wolfe2

            a, *_ = scalar_search_wolfe2(lambda x: oracle.func_directional(x, d_k, a),
                                         lambda x: oracle.grad_directional(x, d_k, a),
                                         oracle.func_directional(0, d_k, a), None,
                                         oracle.grad_directional(0, d_k, a),
                                         self.c1,
                                         self.c2)
            if a:
                return a
            return LineSearchTool(method='Armijo', c1=self.c1, alpha_0=self.alpha_0).line_search(oracle, x_k, d_k,
                                                                                                 previous_alpha)

        elif self._method == 'Armijo':
            return armijo()

        elif self._method == 'Constant':
            return self.c


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000, line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k, a_k = np.copy(x_0), None


    grad_0_norm = np.linalg.norm(oracle.grad(x_0)) ** 2
    stop = tolerance * grad_0_norm
    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)

    try:
        start_time = time.time()
        for it in range(max_iter):

            d_k = oracle.grad(x_k)
            if np.linalg.norm(d_k) ** 2 <= stop:
                if display:
                    print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time, np.linalg.norm(d_k) ** 2))
                if trace:
                    history['time'].append(time.time() - start_time)
                    history['func'].append(oracle.func(x_k))
                    history['grad_norm'].append(np.linalg.norm(d_k))
                    if x_k.size <= 2:
                        history['x'].append(x_k)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, 2 * a_k if a_k else None)
                x_k = x_k - a_k * d_k

            if display:
                print("iteration: {}\ntime: {}\ngrad_norm: {}\n\n".format(it, time.time() - start_time, np.linalg.norm(d_k)))
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(d_k))
                if x_k.size <= 2:
                    history['x'].append(x_k)

        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history



def newton(oracle, x_0, tolerance=1e-9, max_iter=50, line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    a_k = None


    grad_0_norm = np.linalg.norm(oracle.grad(x_0)) ** 2
    stop = tolerance * grad_0_norm

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append(0.0)
        history['grad_norm'].append(grad_0_norm)

    try:
        start_time = time.time()
        for it in range(max_iter):
            if display:
                print(it)
            g_k = oracle.grad(x_k)
            try:
                h_k = oracle.hess(x_k)
                d_k = scipy.linalg.cho_solve(scipy.linalg.cho_factor(h_k), g_k)
            except LinAlgError:
                return x_k, 'newton_direction_error', history

            if np.linalg.norm(d_k) ** 2 <= stop:
                if trace:
                    history['time'].append(time.time() - start_time)
                    history['func'].append(oracle.func(x_k))
                    history['grad_norm'].append(np.linalg.norm(d_k))
                    if x_k.size <= 2:
                        history['x'].append(x_k)
                return x_k, 'success', history
            else:
                a_k = line_search_tool.line_search(oracle, x_k, -d_k, a_k if a_k else None)
                x_k = x_k - a_k * d_k

            # history
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(d_k))
                if x_k.size <= 2:
                    history['x'].append(x_k)
        else:
            return x_k, 'iterations_exceeded', history
    except BaseException as e:
        print(e)
        return x_k, 'computational_error', history