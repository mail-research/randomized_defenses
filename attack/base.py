import torch
import torch.nn.functional as F
import math
from .utils import *

class BaseAttack:
    def __init__(self, model, log, lp, eps, n_queries_each=1):
        self.lp = lp
        self.eps = eps
        self.log = log
        self.model = model
        self.n_queries_each = n_queries_each

    @torch.no_grad()
    def get_prob(self, x, y):
        probs = []
        bsz = 64
        n_batches = math.ceil(x.shape[0] / 64)
        for i in range(n_batches):
            x_batch = x[i*bsz:(i+1)*bsz]
            y_batch = y[i*bsz:(i+1)*bsz]
            logits = self.model(x_batch)
            prob = logits.softmax(dim=1).cpu()
            # out = self.model(x).softmax(dim=-1)
            prob = torch.gather(prob, 1, y_batch.unsqueeze(1)).squeeze(1)
            probs.append(prob)
        # prob = torch.index_select(F.softmax(out, dim=-1), 1, y).cpu()
        return torch.cat(probs)

    # @torch.no_grad()
    # def get_pred(self, x):
    #     preds = []
    #     bsz = 64
    #     n_batches = math.ceil(x.shape[0] / 64)
    #     for i in range(n_batches):
    #         x_batch = x[i*bsz:(i+1)*bsz]
    #         logits = self.model(x_batch)
    #         preds.append(logits.max(1)[1])
    #     return torch.cat(preds)
        # prob = torch.index_select(F.softmax(out, dim=-1), 1, y).cpu()
        # out = self.model.predict(x)
        # return out.max(1)[1]

    @torch.no_grad()
    def get_pred(self, x, stop_criterion, y=None):
        def verify_pred(model, x, n_run):
            preds = []
            for i in range(n_run):
                pred = model.predict(x, torch.is_tensor(x)).argmax(1)
                preds.append(pred)
            preds = torch.stack(preds, dim=1)
            # breakpoint()
            return torch.mode(preds, dim=1)[0]

        if stop_criterion == 'single':
            pred = self.model.predict(x, torch.is_tensor(x)).argmax(1)
            # out = pred != y
        elif stop_criterion == 'without_defense':
            pred = self.model.predict(x, torch.is_tensor(x), defense=False).argmax(1)
            # out = pred != y
        elif stop_criterion == 'fast_exp':
            pred = self.model.predict(x, torch.is_tensor(x)).argmax(1)
            out = pred != y
            if sum(out) > 0:
                pred[out] = verify_pred(self.model, x[out], 9)
        return pred# != y
    
    @torch.no_grad()
    def get_remaining(self, x, stop_criterion, y=None):
        def verify_pred(model, x, n_run, y):
            # preds = []
            score = 0
            for i in range(n_run):
                pred = model.predict(x, torch.is_tensor(x)).argmax(1)
                score += (pred == y).float() * 2 - 1
            return score > 0
            # preds = torch.stack(preds, dim=1)
            # breakpoint()
            # return torch.mode(preds, dim=1)[0]

        if stop_criterion == 'single' or stop_criterion == 'none':
            pred = self.model.predict(x, torch.is_tensor(x)).argmax(1)
            out = pred == y
        elif stop_criterion == 'without_defense':
            pred = self.model.predict(x, torch.is_tensor(x), defense=False).argmax(1)
            out = pred == y
        elif stop_criterion == 'fast_exp':
            pred = self.model.predict(x, torch.is_tensor(x)).argmax(1)
            out = pred == y
            if sum(~out) > 0:
                out[~out] = verify_pred(self.model, x[~out], 9, y[~out])
        return out# != y

    def get_adaptive_output(self, x, M):
        output = 0
        for  _ in range(M):
            output += self.model.predict(x, True)
        output = output / M
        return output


    def get_loss(self, x, y, M=1):
        """
            attacker aims for high loss
        """
        if M > 1:
            pred = self.get_adaptive_output(x, M)
        else:
            pred = self.model.predict(x, return_tensor=True)
        loss = - self.model.loss(y, pred, targeted=self.targeted)
        return torch.tensor(loss, device=x.device)

    def attack(self, x, labels, n_queries, stop_criterion):
        total_queries = 0
        n_ex = x.shape[0]
        n_iter = n_queries // self.n_queries_each 
        last_n_queries = n_queries % self.n_queries_each 
        # queries = torch.zeros(n_ex, device=x.device)
        # pred = self.get_pred(x, stop_criterion, labels)
        remaining = self.get_remaining(x, stop_criterion, labels)
        queries_list = [1] * (remaining.shape[0] - remaining.sum()).item()
        
        base_x = x.clone()
        if self.lp == 'l2':
            projection = get_l2_proj#(x, self.eps)
        elif self.lp == 'linf':
            projection = get_linf_proj#(x, self.eps)
        for i in range(n_iter + 1):
            if stop_criterion == 'none':
                self.idx_to_fool = torch.ones(n_ex).bool()
            else:
                self.idx_to_fool = remaining#pred == labels
                x, labels, base_x = x[self.idx_to_fool], labels[self.idx_to_fool], base_x[self.idx_to_fool]
            if x.shape[0] == 0:
                print('No sample left')
                break
            x, q = self.run_one_iter(x, labels)
            total_queries += q
            x = projection(base_x, x, self.eps)
            # pred = self.get_pred(x, stop_criterion, labels)
            # # breakpoint()
            # acc = (pred == labels).sum() / n_ex
            remaining = self.get_remaining(x, stop_criterion, labels)
            queries_list += [total_queries] * (remaining.shape[0] - remaining.sum()).item()
            # breakpoint()
            acc = remaining.sum() / n_ex
            self.log.print(f'Iter: {i}, Query: {total_queries}, Acc: {acc:.3f}, Avg queries: {np.mean(queries_list):.3f}')
            if total_queries >= n_queries:
                print('Reach limit query')
                break
        return x
            