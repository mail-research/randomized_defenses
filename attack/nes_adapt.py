from .base import BaseAttack
from .utils import *

import numpy as np
import torch

def get_margin(model, x, y, stop_criterion, targeted=False):
    if stop_criterion == 'single':
        logits = model.predict(x, torch.is_tensor(x))
        margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    elif stop_criterion == 'without_defense':
    ## base model
        base_logits = model.predict(x, torch.is_tensor(x), defense=False)
        margin_min = model.loss(y, base_logits, targeted, loss_type='margin_loss')
    elif stop_criterion == 'fast_exp':
    ## fast expectation
        logits = model.predict(x, torch.is_tensor(x))
        margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
        if sum(margin_min <= 0) > 0:
            margin_min[margin_min <= 0] = get_remaining_idx(model, x[margin_min <= 0], y[margin_min <= 0])
    elif stop_criterion == 'exp':
    ## true expectation
        margin_min = get_remaining_idx(model, x, y)
    return margin_min

def get_idx(model, x, y, stop_criterion):
    def verify_pred(model, x, n_run):
        preds = []
        for i in range(n_run):
            pred = model.predict(x, torch.is_tensor(x)).argmax(1)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)
        # breakpoint()
        return torch.mode(preds, dim=1)[0]

    if stop_criterion == 'single':
        pred = model.predict(x, torch.is_tensor(x)).argmax(1)
        out = pred != y
    elif stop_criterion == 'without_defense':
        pred = model.predict(x, torch.is_tensor(x), defense=False).argmax(1)
        out = pred != y
    elif stop_criterion == 'fast_exp':
        pred = model.predict(x, torch.is_tensor(x)).argmax(1)
        out = pred != y
        if sum(out) > 0:
            pred[out] = verify_pred(model, x[out], 9)
    return pred != y

class NES_Adaptive(BaseAttack):

    def __init__(self, model, log, eps, lp, n_queries_each, sigma, step_size, targeted=False, M=1):
        super().__init__(model, log, lp, eps, n_queries_each=n_queries_each)
        self.sigma = sigma
        self.targeted = targeted
        self.step_size = step_size
        self.M = M
        if lp == 'l2':
            self.update_fn = l2_step
        elif lp == 'linf':
            self.update_fn = linf_step

    @torch.no_grad()
    def run_one_iter(self, x, labels, n_queries=None):
        num_dim = len(x.shape[1:])
        total_grad = torch.zeros_like(x)
        n_queries = self.n_queries_each
        for _ in range(n_queries):
            tangent = torch.randn_like(x) #/(np.prod(list(x.shape[1:]))**0.5) 
            forward_x = x + self.sigma * tangent
            backward_x = x - self.sigma * tangent
            # forward_y = self.model.predict(forward_x, True)
            # backward_y = self.model.predict(backward_x, True)
            forward_y = self.get_adaptive_output(forward_x, self.M)
            backward_y = self.get_adaptive_output(backward_x, self.M)
            change = (self.model.loss(labels, forward_y, targeted=self.targeted) - self.model.loss(labels, backward_y, targeted=self.targeted)) / (2 * self.sigma)
            total_grad -= torch.tensor(change, device=tangent.device).reshape(-1, *[1] * num_dim) * tangent

        new_x = self.update_fn(x, total_grad, self.step_size)
        # new_x = x + self.step_size * total_grad.sign()# / total_grad.norm(p=2, dim=list(range(1, num_dim+1)), keepdim=True)
        return new_x, 2 * n_queries * self.M