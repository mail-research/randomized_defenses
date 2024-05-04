from .base import BaseAttack
from .utils import *
import numpy as np

import torch

class SignHunt(BaseAttack):

    def __init__(self, model, log, eps, lp, targeted=False, adaptive=False, M=1):
        super().__init__(model, log, lp, eps)

        self.targeted = targeted
        self.x0 = None
        self.adaptive = adaptive
        self.M = M
        if lp == 'l2':
            self.prior_step = step
            self.update_fn = l2_step
        elif self.lp == 'linf':
            self.prior_step = eg_step
            self.update_fn = linf_step

    def run_one_iter(self, x, labels):
        x_shape = list(x.shape)
        n_dim = np.prod(x_shape[1:])
        add_queries = 0
        if self.x0 is None:
            self.x0 = x.clone()
            self.h = 0
            self.i = 0
        if self.i == 0 and self.h == 0:
            self.sign = torch.ones(x_shape[0], n_dim).sign()
            forward_x = self.update_fn(self.x0, self.sign.view(x_shape), self.eps)
            backward_x = self.x0
            est_deriv = (self.get_loss(forward_x, labels, self.M) - self.get_loss(backward_x, labels, self.M)) / self.eps
            self.best_est_deriv = est_deriv
            add_queries = 2 * self.M
        else:
            self.best_est_deriv = self.best_est_deriv[self.idx_to_fool]
            self.x0 = self.x0[self.idx_to_fool]
            self.sign = self.sign[self.idx_to_fool]

        chunk_len = np.ceil(n_dim / (2 ** self.h)).astype(int)
        istart = self.i * chunk_len
        iend = min(n_dim, (self.i + 1) * chunk_len)
        self.sign[:, istart:iend] *= -1
        forward_x = self.update_fn(self.x0, self.sign.view(x_shape), self.eps)
        backward_x = self.x0
        est_deriv = (self.get_loss(forward_x, labels, self.M) - self.get_loss(backward_x, labels, self.M)) / self.eps
        assert self.best_est_deriv.shape == est_deriv.shape
        self.sign[est_deriv < self.best_est_deriv, istart:iend] *= -1

        self.best_est_deriv[est_deriv >= self.best_est_deriv] = est_deriv[est_deriv >= self.best_est_deriv]

        new_x = self.update_fn(self.x0, self.sign.view(x_shape), self.eps)
        self.i += 1
        if self.i == 2 ** self.h or iend == n_dim:
            self.h += 1
            self.i = 0
            if self.h == np.ceil(np.log2(n_dim)).astype(int) + 1:
                self.x0 = x.clone()
                self.h = 0

        return new_x, add_queries + self.M

        