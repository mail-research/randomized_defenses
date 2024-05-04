from .base import BaseAttack
from .utils import *

import torch

class SimBA(BaseAttack):

    def __init__(self, model, log, eps, img_size, freq_dims, stride, linf_bound=0.05, order='rand', pixel_attack=False):
        super().__init__(model, log, 'l2', eps)
        self.img_size = img_size
        self.freq_dims = freq_dims
        self.stride = stride
        self.linf_bound = linf_bound
        self.order = order
        self.pixel_attack = pixel_attack
        
    def expand_vector(self, x, size):
        bsz = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(bsz, 3, self.img_size, self.img_size)
        z[:, :, :size, :size] = x
        return z


    def attack(self, img, labels, max_iters, stop_criterion, targeted=False):
        bsz = img.shape[0]
        image_size = img.shape[2]
        if self.order == 'rand':
            indices = torch.randperm(3 * self.freq_dims * self.freq_dims)[:max_iters]
        elif self.order == 'diag':
            indices = diagonal_order(image_size, 3)[:max_iters]
        elif self.order == 'strided':
            indices = block_order(image_size, 3, initial_size=self.freq_dims, stride=self.stride)[:max_iters]
            # breakpoint()
        else:
            indices = block_order(image_size, 3)[:max_iters]
        if self.order == 'rand':
            expand_dims = self.freq_dims
        else:
            expand_dims = image_size

        n_dims = 3 * expand_dims * expand_dims
        pert = torch.zeros(bsz, n_dims)
        prob = torch.zeros(bsz, max_iters)
        succ = torch.zeros(bsz, max_iters)
        queries = torch.zeros(bsz, max_iters)
        l2 = torch.zeros(bsz, max_iters)
        linf = torch.zeros(bsz, max_iters)
        prev_prob = self.get_prob(img, labels)
        # pred = self.get_pred(img, stop_criterion, labels)
        if self.pixel_attack:
            trans = lambda z: z#.clip(- self.linf_bound, self.linf_bound)
        else:
            trans = lambda z: block_idct(z, block_size=image_size, linf_bound=self.linf_bound)

        remaining = torch.ones(bsz).bool()
        remaining_indices = torch.arange(0, bsz).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded_pert = trans(self.expand_vector(pert, expand_dims))
            remaining_labels = labels[remaining_indices]
            perturbed_img = (img[remaining_indices] + expanded_pert[remaining_indices]).clamp(0, 1)
            l2[:, k] = expanded_pert.view(bsz, -1).norm(2, 1)
            linf[:, k] = expanded_pert.view(bsz, -1).abs().max(1)[0]
            # pred_next = self.get_pred(perturbed_img, stop_criterion, remaining_labels)
            # pred[remaining_indices] = pred_next
            if targeted:
                # remaining = pred.ne(labels)
                remaining[remaining_indices] = ~self.get_remaining(perturbed_img, stop_criterion, remaining_labels)
            else:
                # remaining = pred.eq(labels)
                remaining[remaining_indices] = self.get_remaining(perturbed_img, stop_criterion, remaining_labels)
            if remaining.sum() == 0:
                adv = (img + expanded_pert).clamp(0, 1)
                probs_k = self.get_prob(adv, labels)
                prob[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                # succ[:, k:] = torch.ones(bsz, max_iters - k)
                queries[:, k:] = torch.zeros(bsz, max_iters - k)
                break
            remaining_indices = torch.nonzero(remaining).squeeze(1)
            # if k > 0:
            #     succ[:, k-1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = self.eps
            left_vec = pert[remaining_indices] + diff
            right_vec = pert[remaining_indices] - diff
            
            adv = (img[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_prob = self.get_prob(adv, labels[remaining_indices])
            queries_k = torch.zeros(bsz)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                left_improved = left_prob > prev_prob[remaining_indices]
            else:
                left_improved = left_prob < prev_prob[remaining_indices]
            
            if left_improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~left_improved]] += 1

            adv = (img[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_prob = self.get_prob(adv, labels[remaining_indices])
            if targeted:
                right_improved = right_prob > torch.max(left_prob, prev_prob[remaining_indices])
            else:
                right_improved = right_prob < torch.min(left_prob, prev_prob[remaining_indices])
            prob_k = prev_prob.clone()

            if left_improved.sum() > 0:
                left_indices = remaining_indices[left_improved]
                pert[left_indices] = left_vec[left_improved, :]
                prob_k[left_indices] = left_prob[left_improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                pert[right_indices] = right_vec[right_improved, :]
                prob_k[right_indices] = right_prob[right_improved]

            prob[:, k] = prob_k
            queries[:, k] = queries_k
            prev_prob = prob[:, k]

            self.log.print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f, l2 = %.2f, linf = %.2f' % (
                        k + 1, queries.sum(1).mean(), prob[:, k].mean(), remaining.float().sum() / 1000, l2[:, k].mean(), linf[:, k].mean()))

        perturbed_img = (img + trans(self.expand_vector(pert, expand_dims))).clamp(0, 1) 
        # pred = self.get_pred(perturbed_img, stop_criterion, labels)
        # if targeted:
        #     # remaining = pred.ne(labels)
        #     remaining = ~self.get_remaining(perturbed_img, stop_criterion, remaining_labels)
        # else:
        #     # remaining = pred.eq(labels)
        #     remaining = self.get_remaining(perturbed_img, stop_criterion, remaining_labels)
        succ[:, max_iters-1] = ~remaining
        return perturbed_img, prob, succ, queries, l2, linf