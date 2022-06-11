import os
import torch
import argparse
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from setup.utils import loaddata, loadmodel, savefile
from sklearn.neighbors import KNeighborsClassifier




def trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=True):
    if use_cuda:
        model = model.cuda()
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr_max'], momentum=0.9, weight_decay=args['weight_decay'])
    train_criterion = nn.CrossEntropyLoss()
    for epoch in range(args['num_epoch']):
        # training
        ave_loss = 0
        step = 0
        for idx, x, target in tqdm(train_loader):
            x, target = to_var(x), to_var(target)
            if args['clean']:
                x_adv = x
            else:
                target_pred = pred_batch(x, model)
                x_adv_init = adv_train(x, target_pred, model, train_criterion, adversary)

                if args['criterion'] == 'angle':
                    angles = compute_angle(args, result_dir, idx, x, x_adv_init)
                    ep = get_ep(angles, args['train_epsilon'], args['criterion'], args['method'], args['exp'], args['threshold'], args['train_ratio'],
                                args['precision'], args['round'])
                    x_adv = adv_train(x, target_pred, model, train_criterion, adversary, ep=ep)
                elif args['criterion'] == 'tan':
                    components = compute_tangent(args, result_dir, idx, x, x_adv_init)
                    ep = get_ep(components, args['train_epsilon'], args['criterion'], args['method'], args['exp'], args['threshold'], args['train_ratio'],
                                args['precision'], args['round'])
                    x_adv = adv_train(x, target_pred, model, train_criterion, adversary, ep=ep)
                else:
                    raise Exception("No such criterion")

            loss = train_criterion(model(x_adv), target)
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1

            model.train()
            lr = lr_schedule(epoch + 1)
            optimizer.param_groups[0].update(lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if (step + 1) % args['print_every'] == 0:
                print("Epoch: [%d/%d], step: [%d/%d], Average Loss: %.4f" %
                      (epoch + 1, args['num_epoch'], step + 1, len(train_loader), ave_loss))
        acc = testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=args['batch_size'])
        print("Epoch {} test accuracy: {:.3f}".format(epoch, acc))
    savefile(args['file_name']+str(round(acc,3)), model, args['dataset'])
    return model




def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)

    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')
    result_dir = args['result_dir']
    model = trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=use_cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument("-n", "--num_epoch", type=int, default=120)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    #parser.add_argument("-l", "--lr", type=float, default=1e-3)
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

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
