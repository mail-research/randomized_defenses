import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import math
import torchvision
import numpy as np
import yaml
from tqdm.auto import tqdm
import timm
from torchvision import datasets as dset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torch.utils.data import DataLoader

from attack.simba import *
from attack.square_attack import *
from attack.bandit import *
from attack.nes import NES
from attack.nes_adapt import NES_Adaptive
from attack.signhunt import SignHunt
from attack.zo_signsgd import ZO_SignSGD
from attack.decision import *
from models.utils import create_robust_model

import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

import math

import argparse
import numpy as np
import os
import utils
np.set_printoptions(precision=5, suppress=True)
os.environ["CURL_CA_BUNDLE"]=""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='Model name.')
    parser.add_argument('--data-path', type=str, default='~/data', help='Directory to data.')
    parser.add_argument('--attack', type=str, default='square_linf')
    parser.add_argument('--exp_folder', type=str, default='exps', help='Experiment folder to store all output.')
    parser.add_argument('--gpu', type=str, default='3', help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                            'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=10000)
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    parser.add_argument('--defense', type=str, default='identical')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--def_position', type=str, default='hidden_feature', help='THe position to insert noise.')
    parser.add_argument('--layer_index', nargs='*', default=-1)
    parser.add_argument('--noise_list', nargs='*', default=[0])
    parser.add_argument('--scale_noise', action='store_true')
    parser.add_argument('--stop_criterion', type=str, default='fast_exp', choices=['single', 'without_defense', 'fast_exp', 'exp', 'none'])
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--num_adapt', type=int, default=1)
    parser.add_argument('--wb', action='store_true')
    args = parser.parse_args()
    try:
        if not isinstance(args.layer_index, int):
            if len(args.layer_index) == 1:
                args.layer_index = int(args.layer_index[0])
            else:
                args.layer_index = [int(_) for _ in args.layer_index]
        args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
        blackbox = True
        device = torch.device(f'cuda:{args.gpu}')
        n_cls = 1000 if args.dataset == 'imagenet' else 10

        if 'torchvision' not in args.model:
            cfg = timm.create_model(args.model).default_cfg
            scale_size = int(math.floor(cfg['input_size'][-2] / cfg['crop_pct']))
            if cfg['interpolation'] == 'bilinear':
                interpolation = torchvision.transforms.InterpolationMode.BILINEAR 
            elif cfg['interpolation'] == 'bicubic':
                interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        else:
            scale_size = 224
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        transform = Compose([
            Resize(scale_size, interpolation=interpolation),
            CenterCrop(size=(cfg['input_size'][-2], cfg['input_size'][-2])),
            ToTensor()
        ])

        if 'square' in args.attack or 'hsja' in args.attack:
            use_numpy = True
        else:
            use_numpy = False
        
        if os.path.exists(f'cache/{args.dataset}.pth'):
            x_test, y_test = torch.load(f'cache/{args.dataset}.pth')
            y_test = y_test.to(int)
        else:
            print(args.dataset)
            if args.dataset == 'cifar10':
                dataset = dset.CIFAR10(args.data_path, train=False, download=True, transform=transform)  
            elif args.dataset == 'imagenet':
                dataset = dset.ImageFolder(args.data_path, transform=transform)

            def get_val_data(dataset, n_cls, n_ex):
                assert n_ex % n_cls == 0
                n_samples_each = n_ex // n_cls
                loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)
                iter_loader = iter(loader)
                img_size = dataset[0][0].shape
                x = [torch.zeros(0, *img_size) for _ in range(n_cls)]
                y = [torch.zeros(0) for _ in range(n_cls)]
                while sum([len(_) != n_samples_each for _ in y]) > 0:
                    img, labels = next(iter_loader)
                    for l in torch.unique(labels):
                        curr_n_ex = y[l].shape[0]
                        if curr_n_ex < n_samples_each:
                            x[l] = torch.cat([x[l], img[labels == l][:n_samples_each - curr_n_ex]])
                            y[l] = torch.cat([y[l], labels[labels == l][:n_samples_each - curr_n_ex]])
                x = torch.cat(x)
                y = torch.cat(y).to(int)
                return x, y
            x_test, y_test = get_val_data(dataset, n_cls, args.n_ex)
            torch.save((x_test, y_test), f'cache/{args.dataset}.pth')

        if use_numpy:
            x_test = x_test.numpy()
            y_test = y_test.numpy()

        config_pth = f'configs/{args.attack}.yaml'
        try:
            with open(config_pth, 'r') as fr:
                config = yaml.safe_load(fr)
        except:
            config = ''

        dataset_name = args.dataset
        if args.noise_list is not None:
            noise_list = [float(_) for _ in args.noise_list]
        
        for noise in noise_list:
            print('Start running for noise scale', noise)
            hps_str = 'model={} defense={} n_ex={} eps={} p={} n_iter={} noise_scale={}'.format(
            args.model, args.defense, args.n_ex, args.eps, args.p, args.n_iter, noise)
            method = args.def_position
            base_dir = 'logs/{}/{}/{}/{}/{}/'.format(args.model, args.exp_folder, args.attack + '_' + str(args.eps) + '_' + str(config), args.dataset, method + f'_layer_{args.layer_index}' + ('_scale_std' if args.scale_noise else ''))
            os.makedirs(base_dir, exist_ok=True)
            log_path = base_dir + f'{hps_str}.log'
        
            log = utils.Logger(log_path)
            log.print(str(args.__dict__))
            log.print(str(config))

            model = create_robust_model(args.model, args.dataset, n_cls, noise, args.defense, args.def_position, device=device, layer_index=args.layer_index, blackbox=blackbox, scale=args.scale_noise)
            print('done load model')
            mean_acc = 0
            for _ in range(5):
                logits_clean = model.predict(x_test, not use_numpy)
                corr_classified = (logits_clean.argmax(1) == y_test)
                acc = corr_classified.float().mean() if torch.is_tensor(corr_classified) else corr_classified.mean()
                mean_acc += acc
            mean_acc /= 5
            # important to check that the model was restored correctly and the clean accuracy is high
            log.print('Clean accuracy: {:.2%}'.format(mean_acc))

            if args.attack == 'simba_dct':
                attacker = SimBA(model, log, args.eps, 224, **config)
                _, prob, succ, queries, l2, linf = attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif args.attack == 'simba_pixel':
                attacker = SimBA(model, log, args.eps, 224, **config)
                _, prob, succ, queries, l2, linf = attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'nes' in args.attack and not 'adapt' in args.attack:
                attacker = NES(model, log, args.eps, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'nes_adapt' in args.attack:
                attacker = NES_Adaptive(model, log, args.eps, M=args.num_adapt, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'bandit' in args.attack:
                attacker = Bandit(model, log, args.eps, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'signhunt' in args.attack:
                attacker = SignHunt(model, log, eps=args.eps, M=args.num_adapt, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'zo_signsgd' in args.attack:
                attacker = ZO_SignSGD(model, log, args.eps, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'square' in args.attack:
                metrics_path = base_dir + 'metrics/'
                os.makedirs(metrics_path, exist_ok=True)
                metrics_path += hps_str + '.metrics'
                square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
                n_queries, x_adv = square_attack(model, x_test, y_test, None, args.eps, args.n_iter,
                                                args.p, metrics_path, args.targeted, args.loss, log, args.stop_criterion, adaptive=args.adaptive, M=args.num_adapt)
                print(f'Noise :{noise}, n queries: {n_queries}')
            elif 'hsja' in args.attack:
                attacker = HSJAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=1, use_numpy=use_numpy, **config)
                bsz = 1
                n_batch = math.ceil(x_test.shape[0] // bsz)
                for i in (pbar := tqdm(range(n_batch))):
                    x = x_test[i * bsz:(i+1) * bsz]
                    y = y_test[i * bsz:(i+1) * bsz]
                    # breakpoint()
                    logs = attacker.run(x, y, model, args.targeted)
                    log.print(str(attacker.result()))
                log.print(str(attacker.result()))
            elif 'rays' in args.attack:
                attacker = RaySAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=1, logger=log, **config)
                attacker.run(x_test, y_test, model, args.targeted)
                log.print(str(attacker.result()))
            elif 'signflip' in args.attack:
                attacker = SignFlipAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=1, logger=log, **config)
                attacker.run(x_test, y_test, model, args.targeted)
                log.print(str(attacker.result()))
            elif 'signopt' in args.attack:
                attacker = SignOPTAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=1, logger=log, **config)
                bsz = 1
                n_batch = math.ceil(x_test.shape[0] // bsz)
                for i in (pbar := tqdm(range(n_batch))):
                    x = x_test[i * bsz:(i+1) * bsz]
                    y = y_test[i * bsz:(i+1) * bsz]
                    logs = attacker.run(x, y, model, args.targeted)
                    log.print(str(attacker.result()))
                log.print(str(attacker.result()))
            elif 'geoda' in args.attack:
                attacker = GeoDAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=1, stop_criterion=args.stop_criterion, **config)
                bsz = 1
                n_batch = math.ceil(x_test.shape[0] // bsz)
                for i in (pbar := tqdm(range(0, n_batch))):
                    x = x_test[i * bsz:(i+1) * bsz]
                    y = y_test[i * bsz:(i+1) * bsz]
                    logs = attacker.run(x, y, model, args.targeted)
                    print(logs[1])
                    log.print(str(attacker.result()))
                log.print(str(attacker.result()))

    except KeyboardInterrupt:
        pass
