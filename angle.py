import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from tangent import compute_angle, compute_tangent
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data


def angle(args, model, result_dir, train_loader, use_cuda=True):
    if use_cuda:
        model = model.cuda()
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])
    train_criterion = nn.CrossEntropyLoss()
    for idx, x, target in tqdm(train_loader):
        x, target = to_var(x), to_var(target)
        if args['clean']:
            x_adv = x
        else:
            target_pred = pred_batch(x, model)
            x_adv_init = adv_train(x, target_pred, model, train_criterion, adversary)

            angles = compute_angle(args, result_dir, idx, x, x_adv_init)

        loss = train_criterion(model(x_adv), target)
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1


def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)

    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')
    result_dir = args['result_dir']
    model = angle(args, model, result_dir, train_loader, use_cuda=use_cuda)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="vgg16")
    parser.add_argument("-n", "--num_epoch", type=int, default=100)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("--criterion", default='angle', choices=['angle', 'tan'])
    parser.add_argument("--method", default='num', choices=['num', 'rank'])
    parser.add_argument("--round", action="store_true", default=False, help='if true, round epsilon vector')
    parser.add_argument("--precision", type=int, default=4, help='precision of rounding the epsilon vector')
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default=r'/data/tangent', help='the working directory that contains AA, AAA')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false", default=True,
                        help="shuffle in training or not")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width', type=int, default=10, help='WRN width factor')
    args = vars(parser.parse_args())
    args['file_name'] = args['file_name'] + '_' + args['criterion'] + '_' + args['method']
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
        args['print_every'] = 300
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.01
        args['num_k'] = 7
        args['epsilon'] = 8 / 255
        args['batch_size'] = 100
        args['print_every'] = 250
    elif args['dataset'] == 'stl10':
        args['alpha'] = 0.0156
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['batch_size'] = 64
        args['print_every'] = 50
    elif args['dataset'] == 'tiny':
        args['alpha'] = 0.002
        args['num_k'] = 10
        args['epsilon'] = 0.01
        args['batch_size'] = 128
        args['print_every'] = 500
        args['num_gpu'] = 2
    else:
        print('invalid dataset')
    print(args)
    main(args)
