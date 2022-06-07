#import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from tqdm import tqdm
#import torchvision
#from torchvision import transforms
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data, GA_PGD

from GAIR import GAIR
#from utils import Logger


'''

# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.epochs >= 110:
            # Train Wide-ResNet
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            elif t / args.epochs < (11 / 12):
                return args.lr_max / 100.
            else:
                return args.lr_max / 200.
        else:
            # Train ResNet
            if t / args.epochs < 0.3:
                return args.lr_max
            elif t / args.epochs < 0.6:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                      [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t // (args.epochs // 10)) * (args.lr_max / 10)
elif args.lr_schedule == 'cosine':
    def lr_schedule(t):
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))



# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
'''




# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch):
    Lam = float(args['Lambda'])
    if args['num_epoch'] >= 110:
        # Train Wide-ResNet
        Lambda = args['Lambda_max']
        if args['Lambda_schedule'] == 'linear':
            if epoch >= 60:
                Lambda = args['Lambda_max'] - (epoch / args['num_epoch']) * (args['Lambda_max'] - Lam)
        elif args['Lambda_schedule'] == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam - 1.0
            elif epoch >= 110:
                Lambda = Lam - 1.5
        elif args['Lambda_schedule'] == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args['Lambda_max']
        if args['Lambda_schedule'] == 'linear':
            if epoch >= 30:
                Lambda = args['Lambda_max'] - (epoch / args['num_epoch']) * (args['Lambda_max'] - Lam)
        elif args['Lambda_schedule'] == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam - 2.0
        elif args['Lambda_schedule'] == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda


'''
# Resume
title = 'GAIRAT'
best_acc = 0
start_epoch = 0


if resume:
    # Resume directly point to checkpoint.pth.tar
    print ('==> GAIRAT Resuming from checkpoint ..')
    print(resume)
    assert os.path.isfile(resume)
    out_dir = os.path.dirname(resume)
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['test_pgd20_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> GAIRAT')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])
'''



def trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=True):
    if use_cuda:
        model = model.cuda()
        #model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr_max'],momentum=0.9, weight_decay=args['weight_decay'])
    train_criterion = nn.CrossEntropyLoss()

    for epoch in range(args['num_epoch']):
        # training
        ave_loss = 0
        step = 0
        # Get lambda
        Lambda = adjust_Lambda(epoch + 1)
        #num_data = 0
        #train_robust_loss = 0
        print(train_loader)
        for idx, (x, target) in enumerate(train_loader):
            x, target = to_var(x), to_var(target)

            x_adv, Kappa = GA_PGD(model, x, target, args['epsilon'], args['alpha'], args['num_k'],
                                         loss_fn="cent",
                                         category="Madry", rand_init=True)

            model.train()
            lr = lr_schedule(epoch + 1)
            optimizer.param_groups[0].update(lr=lr)
            optimizer.zero_grad()

            if (epoch + 1) >= args['begin_epoch']:
                Kappa = Kappa.cuda()
                loss = train_criterion(model(x_adv),target)
                # Calculate weight assignment according to geometry value
                normalized_reweight = GAIR(args['num_k'], Kappa, Lambda, args['weight_assignment_function'])
                loss = loss.mul(normalized_reweight).mean()
            else:
                loss = train_criterion(model(x_adv),target)

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
    acc = float(correct_cnt.double()/total_cnt)
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

def GA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa


def perturb3(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for i in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta


        diff = x_adv - data

        diff.clamp_(-epsilon, epsilon)

        x_adv.detach().copy_((diff + data).clamp_(0, 1))
    return x_adv

def perturb2(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for i in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)

        loss_adv.backward()
        grad = x_adv.grad
        x_adv = x_adv + step_size * grad.sign()

        diff = x_adv - data

        diff.clamp_(-epsilon, epsilon)

        x_adv.detach().copy_((diff + data).clamp_(0, 1))
    return x_adv


def perturb1(model, data, target, epsilon,step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    y_var = to_var(target)

    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_steps):
        x_adv = to_var(x_adv, requires_grad=True)

        scores = model(x_adv)
        loss = loss_fn(scores, y_var)
        loss.backward()
        grad = x_adv.grad
        x_adv = x_adv+ step_size * grad.sign()

        diff = x_adv - data

        diff.clamp_(-epsilon, epsilon)

        x_adv.detach().copy_((diff + data).clamp_(0, 1))
    return x_adv

def perturb0(model, data, target, epsilon,step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()

    rand = torch.Tensor(data.shape).uniform_(-epsilon, epsilon)
    if torch.cuda.is_available():
        rand = rand.cuda()
    x_adv = data + rand

    loss_fn = nn.CrossEntropyLoss()

    y_var = to_var(target)

    for i in range(num_steps):
        x_adv = to_var(x_adv, requires_grad=True)

        scores = model(x_adv)
        loss = loss_fn(scores, y_var)
        loss.backward()
        grad = x_adv.grad
        x_adv = x_adv + step_size * grad.sign()

        diff = x_adv - data

        diff.clamp_(-epsilon, epsilon)
        x_adv.detach().copy_((diff + data).clamp_(0, 1))
    return x_adv


def eval_robust0(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv = perturb0(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category,
                             rand_init=random)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy



def eval_robust1(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv = perturb1(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

def eval_robust2(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv = perturb2(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

def eval_robust3(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            x_adv = perturb3(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

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
    # Setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])


    trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')
    result_dir = args['result_dir']
    model = trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=use_cuda)
    testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=args['batch_size'])
    testattack(model, test_loader, args, use_cuda=use_cuda)

    test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=7, epsilon=0.031, step_size=0.007,
                                           loss_fn="cent", category="Madry", random=True)


    test_pgd20_acc3 = eval_robust3(model, test_loader, perturb_steps=7, epsilon=0.031, step_size=0.007,
                                           loss_fn="cent", category="Madry", random=True)
    test_pgd20_acc2 = eval_robust2(model, test_loader, perturb_steps=7, epsilon=0.031, step_size=0.007,
                                           loss_fn="cent", category="Madry", random=True)
    test_pgd20_acc1 = eval_robust1(model, test_loader, perturb_steps=7, epsilon=0.031, step_size=0.007,
                                           loss_fn="cent", category="Madry", random=True)
    test_pgd20_acc0 = eval_robust0(model, test_loader, perturb_steps=7, epsilon=0.031, step_size=0.007,
                                           loss_fn="cent", category="Madry", random=True)
    print(test_pgd20_acc)
    print(test_pgd20_acc0)
    print(test_pgd20_acc1)
    print(test_pgd20_acc2)
    print(test_pgd20_acc3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAIRAT: Geometry-aware instance-dependent adversarial training')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-n", "--num_epoch", type=int, default=100)
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    #parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    #parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
    #parser.add_argument('--step-size', type=float, default=0.007, help='step size')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--random', type=bool, default=True,
                        help="whether to initiat adversarial sample with random noise")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width', type=int, default=10, help='WRN width factor')
    parser.add_argument('--drop-rate', type=float, default=0.0, help='WRN drop rate')
    #parser.add_argument('--resume', type=str, default=None, help='whether to resume training')
    #parser.add_argument('--out-dir', type=str, default='./GAIRAT_result', help='dir of output')
    parser.add_argument('--lr-schedule', default='piecewise',
                        choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument('--Lambda', type=str, default='-1.0', help='parameter for GAIR')
    parser.add_argument('--Lambda_max', type=float, default=float('inf'), help='max Lambda')
    parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
    parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete', 'Sigmoid', 'Tanh'])
    parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')

    # parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')

    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default=r'/data/tangent', help='the working directory that contains AA, AAA')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false", default=True,
                        help="shuffle in training or not")
    args = vars(parser.parse_args())
    #args['file_name'] = args['file_name'] + '_' + args['criterion'] + '_' + args['method']

    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
        args['print_every'] = 300
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.007
        args['num_k'] = 7
        args['epsilon'] = 0.031 #8 / 255
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
    #momentum = args.momentum
    #weight_decay = args.weight_decay
    #resume = args.resume
    #out_dir = args.out_dir

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
