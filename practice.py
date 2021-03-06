import os
import torch
import torch.nn as nn
import torch
import argparse
import math
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from tangent import compute_angle, compute_tangent
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data, GA_PGD
from setup.setup_wrn import wrn
from numpy import linalg as LA
from scipy.stats import rankdata

def distance(x_adv, X_train):
    X_train = X_train.reshape(50000,-1)
    X_train = X_train.cpu().numpy()
    x_adv = x_adv.reshape(1,-1)
    dist = []
    for i in range(50000):
        dist = np.append(dist, LA.norm(X_train[i,:] - x_adv[0]))
    return dist


def check(classifier, train_loader, test_loader, args, use_cuda=True):

    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    X_train, _ = load_CIFAR10(train_loader)
    if use_cuda:
        X_train = X_train.cuda()

    # load the model from disk
    filename = './models/finalized_knn.sav'
    knn = pickle.load(open(filename, 'rb'))

    pbar = tqdm(test_loader)
    for X, y in pbar:
        X_adv = adversary.perturb(X, y)
        X_adv_knn = X_adv.cpu().numpy()
        X_adv_knn = np.reshape(X_adv_knn, (X_adv_knn.shape[0], -1))

        predict_idx = knn.predict(X_adv_knn)
        print(X_adv_knn[[0]])

        dist = distance(X_adv_knn[[0]], X_train)
        print(dist)
        print(np.mean(dist))
        print(np.median(dist))
        print(np.amin(dist))
        print(dist[predict_idx[0]])

        rank = rankdata(dist)
        print(rank[predict_idx[0]])
        print(dist.shape)

    pbar.close()

def load_CIFAR10(train_loader):
    i = 0
    for idx, x, target in tqdm(train_loader):
        if i == 0:
            X_train = x
            idx_train = idx
            i += 1
        else:
            X_train = torch.cat([X_train, x], axis=0)
            idx_train = torch.cat([idx_train, idx], axis=0)
    return X_train, idx_train

def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)

    depth = args['depth']
    width = args['width']
    model = wrn(depth=depth, num_classes=10, widen_factor=width, dropRate=0)

    model.load_state_dict(torch.load(os.path.join('./models/cifar10/cifar10_adapt_tan_num0.894')))
    if use_cuda:
        model = model.cuda()

    check(model, train_loader, test_loader, args, use_cuda=use_cuda)
    #testattack(model, test_loader, args, use_cuda=use_cuda)
    #test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4, loss_fn="cent",
    #            category="Madry", random=True)
    #print(test_pgd20_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument("-n", "--num_epoch", type=int, default=120)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--lr-schedule', default='piecewise',
                        choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument("--criterion", default='angle', choices=['angle', 'tan'])
    parser.add_argument("--method", default='num', choices=['num', 'rank','skip','rank_binary','rank_exp','num_exp'])
    parser.add_argument("--round", action="store_true", default=False, help='if true, round epsilon vector')
    parser.add_argument("--precision", type=int, default=4, help='precision of rounding the epsilon vector')
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default=r'/data/tangent', help='the working directory that contains AA, AAA')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--standard", action="store_true", default=False, help='if true, standard adversarial training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false", default=True,
                        help="shuffle in training or not")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width', type=int, default=10, help='WRN width factor')
    parser.add_argument('--threshold', type=float, default=0.4, help='adaptive train threshold')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='adaptive train ratio')
    parser.add_argument('--train_epsilon', type=float, default=0.031, help='adaptive train ratio')
    parser.add_argument('--exp', type=float, default=2, help='criterion exponent')
    args = vars(parser.parse_args())
    args['file_name'] = args['file_name'] + '_' + args['criterion'] + '_' + args['method']

    args['alpha'] = 0.01
    args['num_k'] = 7
    args['epsilon'] = 8 / 255
    args['batch_size'] = 100
    args['print_every'] = 250

    print(args)


    # Training settings
    seed = args['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
