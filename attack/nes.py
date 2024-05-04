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

class NES(BaseAttack):

    def __init__(self, model, log, eps, lp, n_queries_each, sigma, step_size, targeted=False):
        super().__init__(model, log, lp, eps, n_queries_each=n_queries_each)
        self.sigma = sigma
        self.targeted = targeted
        self.step_size = step_size
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
            forward_y = self.model.predict(forward_x, True)
            backward_y = self.model.predict(backward_x, True)
            change = (self.model.loss(labels, forward_y, targeted=self.targeted) - self.model.loss(labels, backward_y, targeted=self.targeted)) / (2 * self.sigma)
            total_grad -= torch.tensor(change, device=tangent.device).reshape(-1, *[1] * num_dim) * tangent

        new_x = self.update_fn(x, total_grad, self.step_size)
        # new_x = x + self.step_size * total_grad.sign()# / total_grad.norm(p=2, dim=list(range(1, num_dim+1)), keepdim=True)
        return new_x, 2 * n_queries #* torch.ones(x.shape[0])

#     def attack(self, x, labels, n_queries, stop_criterion):
#         total_queries = 0
#         n_ex = x.shape[0]
#         n_iter = n_queries // self.n_queries_each 
#         last_n_queries = n_queries % self.n_queries_each 
#         # queries = torch.zeros(n_ex, device=x.device)
#         margin = get_margin(self.model, x, labels, stop_criterion)
#         base_x = x.clone()
#         if self.lp == 'l2':
#             projection = get_l2_proj#(x, self.eps)
#         elif self.lp == 'linf':
#             projection = get_linf_proj#(x, self.eps)
#         for i in range(n_iter + 1):
            
#             idx_to_fool = margin > 0
#             x, labels, base_x = x[idx_to_fool], labels[idx_to_fool], base_x[idx_to_fool]
#             x, q = self.run_one_iter(x, labels, self.n_queries_each // 2 if i < n_iter else last_n_queries // 2)
#             total_queries += q
#             x = projection(base_x, x, self.eps)
#             margin = get_margin(self.model, x, labels, stop_criterion)
#             acc = (margin > 0).sum() / n_ex
#             self.log.print(f'Iter: {i}, Query: {total_queries}, Acc: {acc}')
            
