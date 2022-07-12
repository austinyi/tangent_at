import os
# os.chdir(r'D:\yaoli\tangent')
import torch
import argparse
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from tangent import compute_angle, compute_tangent
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data, GA_PGD


def get_ep(inputs, epsilon, criterion, method, exp, threshold=0.4, ratio=0.5, precision=3, rou=True):
    cri_method = criterion + '_' + method
    if cri_method == 'angle_num':
        ep = (1 / (inputs * np.max(1 / inputs))) * epsilon
    elif cri_method == 'tan_num':
        ep = inputs / np.max(inputs) * epsilon
    elif cri_method == 'tan_num2':
        ep = np.min(inputs) / inputs * epsilon
    elif cri_method == 'angle_rank':
        rank = np.argsort(
            np.argsort(1 / inputs)) + 1  # to remove zero, 1/inputs since for angle the smaller the larger the epsilon
        ep = rank / inputs.shape[0] * epsilon
    elif cri_method == 'tan_rank':
        rank = np.argsort(np.argsort(inputs)) + 1
        ep = rank / inputs.shape[0] * epsilon
    elif cri_method == 'angle_rank2':
        rank = np.argsort(np.argsort(inputs)) + 1
        ep = rank / inputs.shape[0] * epsilon
    elif cri_method == 'tan_rank2':
        rank = np.argsort(
            np.argsort(1 / inputs)) + 1  # to remove zero, 1/inputs since for angle the smaller the larger the epsilon
        ep = rank / inputs.shape[0] * epsilon
    elif cri_method == 'angle_skip':
        ep = np.zeros(inputs.size)
        ep[inputs < threshold*math.pi] = epsilon
    elif cri_method == 'tan_skip':
        ep = np.zeros(inputs.size)
        ep[inputs > threshold] = epsilon
    elif cri_method == 'angle_rank_binary':
        ep = np.zeros(inputs.size)
        rank = np.argsort(
            np.argsort(1 / inputs)) + 1
        cri = int(inputs.size*ratio)
        ep[rank >= cri] = epsilon
    elif cri_method == 'tan_rank_binary':
        ep = np.zeros(inputs.size)
        rank = np.argsort(np.argsort(inputs)) + 1
        cri = int(inputs.size * ratio)
        ep[rank >= cri] = epsilon
    elif cri_method == 'angle_rank_exp':
        rank = np.argsort(
            np.argsort(1 / inputs)) + 1  # to remove zero, 1/inputs since for angle the smaller the larger the epsilon
        ep = np.power(rank / inputs.shape[0], exp) * epsilon
    elif cri_method == 'tan_rank_exp':
        rank = np.argsort(np.argsort(inputs)) + 1
        ep = np.power(rank / inputs.shape[0], exp) * epsilon
    elif cri_method == 'angle_num_exp':
        ep = np.power((1 / (inputs * np.max(1 / inputs))), exp) * epsilon
    elif cri_method == 'tan_num_exp':
        ep = np.power(inputs / np.max(inputs), exp) * epsilon
    elif cri_method == 'tan_random':
        # ep = np.random.rand(inputs.shape[0])*epsilon
        ep = (np.arange(0, inputs.shape[0]) + 1) / inputs.shape[0] * epsilon
        np.random.shuffle(ep)
    else:
        raise Exception("No such criterion method combination")
    if rou:
        ep = np.round(ep, precision)
    return ep

def reweightedLoss(logs, targets, ep):
    out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        out[i] = logs[i][targets[i]]*ep[i]/0.031
    return -out.sum()/len(out)


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
            target_pred = pred_batch(x, model)
            x_adv = adv_train(x, target_pred, model, train_criterion, adversary)

            if args['criterion'] == 'angle':
                angles = compute_angle(args, result_dir, idx, x, x_adv)
                ep = get_ep(angles, args['train_epsilon'], args['criterion'], args['method'], args['exp'],
                            args['threshold'], args['train_ratio'], args['precision'], args['round'])
            elif args['criterion'] == 'tan':
                components = compute_tangent(args, result_dir, idx, x, x_adv)
                ep = get_ep(components, args['train_epsilon'], args['criterion'], args['method'], args['exp'],
                            args['threshold'], args['train_ratio'], args['precision'], args['round'])
            else:
                raise Exception("No such criterion")

            log_softmax = torch.nn.LogSoftmax(dim=1)
            x_log = log_softmax(model(x_adv))
            loss = reweightedLoss(x_log, target, ep)

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
        # savefile(args['file_name']+str(round(acc,3)), model, args['dataset'])
    return model

def testClassifier(test_loader, model, use_cuda=True, batch_size=100):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double() / total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc


def testattack(classifier, test_loader, args, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=args['epsilon'], k=20, a=args['alpha'])
    param = {
        'test_batch_size': args['batch_size'],
        'epsilon': args['epsilon'],
    }
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv, _ = GA_PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)

    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')
    result_dir = args['result_dir']
    model = trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=use_cuda)
    testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=args['batch_size'])
    testattack(model, test_loader, args, use_cuda=use_cuda)
    test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4, loss_fn="cent",
                category="Madry", random=True)
    print(test_pgd20_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument("-n", "--num_epoch", type=int, default=120)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--lr-schedule', default='piecewise',
                        choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument("--criterion", default='angle', choices=['angle', 'tan'])
    parser.add_argument("--method", default='num', choices=['num', 'num2', 'rank','rank2','skip','rank_binary','rank_exp','num_exp','random'])
    parser.add_argument("--round", action="store_true", default=False, help='if true, round epsilon vector')
    parser.add_argument("--precision", type=int, default=4, help='precision of rounding the epsilon vector')
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default=r'/data/tangent', help='the working directory that contains AA, AAA')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--standard", action="store_true", default=False, help='if true, standard adversarial training')
    parser.add_argument("--save", action="store_true", default=False, help='if true, save the trained model')
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
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
        args['print_every'] = 300
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 2 / 255
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

    # Learning schedules
    if args['lr_schedule'] == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args['num_epoch'] * 2 // 5, args['num_epoch']], [0, args['lr_max'], 0])[0]
    elif args['lr_schedule'] == 'piecewise':
        def lr_schedule(t):
            if args['num_epoch'] >= 110:
                # Train Wide-ResNet
                if t / args['num_epoch'] < 0.5:
                    return args['lr_max']
                elif t / args['num_epoch'] < 0.75:
                    return args['lr_max'] / 10.
                elif t / args['num_epoch'] < (11 / 12):
                    return args['lr_max'] / 100.
                else:
                    return args['lr_max'] / 200.
            else:
                # Train ResNet
                if t / args['num_epoch'] < 0.3:
                    return args['lr_max']
                elif t / args['num_epoch'] < 0.6:
                    return args['lr_max'] / 10.
                else:
                    return args['lr_max'] / 100.
    elif args['lr_schedule'] == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args['num_epoch'] // 3, args['num_epoch'] * 2 // 3, args['num_epoch']],
                                          [args['lr_max'], args['lr_max'], args['lr_max'] / 10, args['lr_max'] / 100])[0]
    elif args['lr_schedule'] == 'onedrop':
        def lr_schedule(t):
            if t < args['lr_drop_epoch']:
                return args['lr_max']
            else:
                return args['lr_one_drop']
    elif args['lr_schedule'] == 'multipledecay':
        def lr_schedule(t):
            return args['lr_max'] - (t // (args['num_epoch'] // 10)) * (args['lr_max'] / 10)
    elif args['lr_schedule'] == 'cosine':
        def lr_schedule(t):
            return args['lr_max'] * 0.5 * (1 + np.cos(t / args['num_epoch'] * np.pi))

    # Training settings
    seed = args['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
