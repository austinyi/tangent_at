import os
import torch
import torch.nn as nn
import torch
import argparse
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from tangent import compute_angle, compute_tangent
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data, GA_PGD
from setup.setup_wrn import wrn


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
    adversary = LinfPGDAttack(classifier, epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    param = {
        'test_batch_size': args['batch_size'],
        'epsilon': args['epsilon'],
    }
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc

def detect_angle(classifier, train_loader, test_loader, args):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    X_train, _ = load_CIFAR10(train_loader)
    if torch.cuda.is_available():
        X_train = X_train.cuda()

    filename = './models/finalized_knn.sav'

    # load the model from disk
    knn = pickle.load(open(filename, 'rb'))

    correct = []
    wrong = []

    '''
    total_correct = 0
    total_samples = len(test_loader.dataset)
    ntested = 0
    '''
    pbar = tqdm(test_loader)
    for X, y in pbar:
        X_adv = adversary.perturb(X, y)
        X_adv_knn = X_adv.cpu().numpy()
        X_adv_knn = np.reshape(X_adv_knn, (X_adv_knn.shape[0], -1))
        predict_idx = knn.predict(X_adv_knn)

        y_pred_adv = pred_batch(X_adv, classifier)
        corr_idx = y_pred_adv.numpy() == y.numpy()

        angles = compute_angle(args, args['result_dir'], predict_idx, X_train[predict_idx], X_adv)
        #print(corr_idx)
        #print(np.invert(corr_idx))
        #print(angles)

        correct = np.append(correct, angles[corr_idx])
        wrong = np.append(wrong, angles[np.invert(corr_idx)])

        #print(np.mean(angles[corr_idx]))
        #print(np.mean(angles[np.invert(corr_idx)]))

        '''
        ntested += y.size()[0]
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()
        pbar.set_postfix(adv_acc="{0}/{1} {2:-6.2f}%".format(total_correct, ntested,
                                                             total_correct*100.0/ntested),
                         refresh=False)
        '''
    pbar.close()

    np.save('./models/correct.npy', correct)
    np.save('./models/wrong.npy', wrong)

    #return correct_angle, wrong_angle


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

    detect_angle(model, train_loader, test_loader, args, use_cuda=use_cuda)
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
