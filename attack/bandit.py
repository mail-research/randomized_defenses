from .base import BaseAttack
from .utils import *

import torch

class Bandit(BaseAttack):

    def __init__(self, model, log, eps, lp, prior_size, step_size, fd_eta, exp_step_size, prior_step_size, targeted=False):
        super().__init__(model, log, lp, eps)

        self.targeted = targeted
        self.step_size = step_size
        self.fd_eta = fd_eta
        self.prior_size = prior_size
        self.prior = None
        self.exp_step_size = exp_step_size
        self.prior_step_size = prior_step_size
        self.upsample = torch.nn.Upsample(size=(224, 224))
        if lp == 'l2':
            self.prior_step = step
            self.update_fn = l2_step
        elif lp == 'linf':
            self.prior_step = eg_step
            self.update_fn = linf_step

    @torch.no_grad()
    def run_one_iter(self, x, labels):
        if self.prior_size is None:
            prior_shape = list(x.shape)
        else:
            prior_shape = list(x.shape)[:-2] + [self.prior_size] * 2 
        if self.prior is None:
            self.prior = torch.zeros(prior_shape) #* 0.5
        else:
            self.prior = self.prior[self.idx_to_fool]
        exp_noise = self.exp_step_size * torch.randn(prior_shape) / (np.prod(list(prior_shape[1:]))**0.5) 

        q1 = self.prior + exp_noise #* self.exp_step_size 
        q2 = self.prior - exp_noise #* self.exp_step_size
        if self.prior_size is not None:
            q1 = self.upsample(q1)
            q2 = self.upsample(q2)
        l1 = self.get_loss(l2_step(x, q1, self.fd_eta), labels)
        l2 = self.get_loss(l2_step(x, q2, self.fd_eta), labels)
        est_deriv = (l1 - l2) / (self.fd_eta * self.exp_step_size)
        est_grad = - est_deriv.view(-1, 1, 1, 1) * exp_noise
        self.prior = self.prior_step(self.prior, est_grad, self.prior_step_size)
        if self.prior_size is None:
            g = self.prior.clone()
        else:
            g = self.upsample(self.prior)
        # breakpoint()

        new_x = self.update_fn(x, g, self.step_size)
        return new_x, 2