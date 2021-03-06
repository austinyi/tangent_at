import os
# os.chdir(r'D:\yaoli\tangent')
import torch
import argparse
import math
import numpy as np
from numpy import linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from setup.utils import savefile
from setup.setup_pgd_adaptive import to_var
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Generate data from half-sphere
# Generate n data which has k classes
def sample_sphere(n=100, k=4, r=1):

    theta = np.random.rand(n) * 2 * math.pi
    phi = np.random.rand(n) * math.pi / 2

    # Generate y
    y = np.zeros(n)
    for i in range(k):
        y[(2 * i * math.pi / k < theta) & (theta < 2 * (i + 1) * math.pi / k)] = i

    # Generate x
    x1 = np.cos(phi) * np.cos(theta)
    x2 = np.cos(phi) * np.sin(theta)
    x3 = np.sin(phi)
    x = np.stack((x1, x2, x3), axis=1)

    x, y = torch.from_numpy(x), torch.from_numpy(y)

    return (x*r).float(), y.long()

# Compute angle between the tangent space of x and x_adv - x
def compute_angle(x, x_adv):
    Angles = []
    for i in range(x.shape[0]):
        angle = math.pi/2 - np.arccos(abs(np.dot(x[i,:], x_adv[i,:] - x[i,:])) / (LA.norm(x[i,:]) * LA.norm(x_adv[i,:] - x[i,:])))
        Angles.append(angle)
    return np.array(Angles)

def cwloss(output, target,confidence=50, num_classes=4):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

# Generate Adversarial Image
def adv(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, num_classes=4, ep=None):
    model.eval()
    '''
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    '''
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()

    if rand_init:
        rand = torch.Tensor(data.shape).uniform_(-epsilon, epsilon)
        if torch.cuda.is_available():
            rand = rand.cuda()
        X = data + rand
    else:
        X = data.clone()

    target = to_var(target)

    for k in range(num_steps):
        x_adv = to_var(X, requires_grad=True)
        #x_adv.requires_grad_()
        output = model(x_adv)

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()

        # Update adversarial data
        x_adv = x_adv.detach() + eta

        diff = x_adv - data
        if ep is not None:
            new_diff = []
            for j in range(ep.shape[0]):
                new_diff.append(torch.clamp(diff[j], -ep[j], ep[j]))
            new_diff = torch.stack(new_diff)
            diff = new_diff
        else:
            diff.clamp_(-epsilon,epsilon)

        x_adv.detach().copy_((diff + data).clamp_(-1, 1))
        #x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        #x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #x_adv = Variable(x_adv, requires_grad=False)
    return x_adv

def get_ep(inputs, epsilon, criterion, method, threshold=0.4, ratio=0.5, precision=3, rou=True):
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
    elif cri_method == 'angle_rank_binary2':
        ep = np.zeros(inputs.size)
        rank = np.argsort(
            np.argsort(1 / inputs)) + 1
        cri = int(inputs.size * ratio)
        ep[rank < cri] = epsilon
    elif cri_method == 'tan_random':
        # ep = np.random.rand(inputs.shape[0])*epsilon
        ep = (np.arange(0, inputs.shape[0]) + 1) / inputs.shape[0] * epsilon
        np.random.shuffle(ep)
    else:
        raise Exception("No such criterion method combination")
    if rou:
        ep = np.round(ep, precision)
    return ep

def trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=True):
    if use_cuda:
        model = model.cuda()
    #adversary = LinfPGDAttack(epsilon=args['train_epsilon'], k=args['num_k'], a=args['alpha'])
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
            elif args['standard']:
                x_adv = adv(model, x, target, epsilon=args['epsilon'], step_size=args['alpha'], num_steps=7, loss_fn="cent", num_classes=args['k'],
                            category="Madry", rand_init=True)
            else:
                x_adv_init = adv(model, x, target, epsilon=args['epsilon'], step_size=args['alpha'], num_steps=7, loss_fn="cent", num_classes=args['k'],
                            category="Madry", rand_init=True)

                if args['criterion'] == 'angle':
                    angles = compute_angle(x, x_adv_init)
                    ep = get_ep(angles, args['epsilon'], args['criterion'], args['method'], args['threshold'], args['train_ratio'],
                                args['precision'], args['round'])
                    #print(angles)
                    x_adv = adv(model, x, target, epsilon=args['epsilon'], step_size=args['alpha'], num_steps=7, loss_fn="cent", num_classes=args['k'],
                        category="Madry", rand_init=True, ep=ep)
                elif args['criterion'] == 'tan':
                    components = compute_tangent(args, result_dir, idx, x, x_adv_init)
                    #ep = get_ep(components, args['train_epsilon'], args['criterion'], args['method'], args['exp'], args['threshold'], args['train_ratio'],
                    #            args['precision'], args['round'])
                    #x_adv = adv_train(x, target, model, train_criterion, adversary, ep=ep)
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

    if args['save']:
        savefile(args['file_name'] + str(round(acc, 3)), model, args['dataset'])
    return model


def testClassifier(test_loader, model, use_cuda=True, batch_size=100):
    model.eval()
    correct_cnt = 0
    total_cnt = 0

    for _, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = to_var(x), to_var(target)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()

    acc = float(correct_cnt / total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc

def save_correct_incorrect_idx(test_loader, model, perturb_steps, epsilon, step_size, loss_fn, category, random, clean=False, use_cuda=True, batch_size=100):
    model.eval()
    correct = []
    incorrect = []
    for _, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = to_var(x), to_var(target)
        if not clean:
            x = adv(model, x, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=random,
                    num_classes=args['k'])
        out = model(x)
        _, pred_label = torch.max(out.data, 1)

        corr_idx = pred_label.numpy() == target.data.numpy()

        for i in range(len(corr_idx)):
            #print(x[i].cpu().detach().numpy())
            #print(correct)
            #print(x[i].cpu().detach().numpy())
            if corr_idx[i]:
                correct.append(x[i].cpu().detach().numpy().tolist())
            else:
                incorrect.append(x[i].cpu().detach().numpy().tolist())
    return correct, incorrect

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random, use_cuda):
    model.eval()
    correct = 0
    with torch.enable_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            x_adv = adv(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random,num_classes=args['k'])
            output = model(x_adv)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

class HalfSphereDataSet(Dataset):
  """
  This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
  getting bogged down by the preprocessing
  """
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]

    return index, _x, _y

class TestHalfSphereDataSet(HalfSphereDataSet):
    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y

def load_data_sphere(n_train=500, n_test=100, k=4, batch_size=32):
    x_train, y_train = sample_sphere(n=n_train, k=k)
    x_test, y_test = sample_sphere(n=n_test, k=k)
    train_loader = DataLoader(HalfSphereDataSet(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TestHalfSphereDataSet(x_test, y_test), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# Define an MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = load_data_sphere(n_train=args['n_train'], n_test=args['n_test'], k=args['k'], batch_size=args['batch_size'])

    print('==> Loading model..')
    model = MLP(output_dim=args['k'])

    print('==> Training starts..')
    result_dir = args['result_dir']
    model = trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=use_cuda)
    testClassifier(test_loader, model, use_cuda=use_cuda, batch_size=args['batch_size'])
    #testattack(model, test_loader, args, use_cuda=use_cuda)
    test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=args['epsilon'], step_size=args['epsilon']/4, loss_fn="cent",
                category="Madry", random=True, use_cuda=use_cuda)
    print(test_pgd20_acc)

    correct, incorrect = save_correct_incorrect_idx(test_loader, model, perturb_steps=20, epsilon=args['epsilon'], step_size=args['epsilon']/4, loss_fn="cent",
                category="Madry", random=True, clean=False, use_cuda=use_cuda, batch_size=args['batch_size'])

    x = np.concatenate((correct, incorrect), axis=0)
    y = np.concatenate((np.ones(len(correct)),np.zeros(len(incorrect))))
    a = x[:, 0]
    b = x[:, 1]
    c = x[:, 2]
    ax = plt.axes(projection='3d')
    ax.scatter(a, b, c, c=y, cmap='viridis', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    #parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="vgg16")
    parser.add_argument("-n", "--num_epoch", type=int, default=50)
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--lr-schedule', default='piecewise',
                        choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument("--criterion", default='angle', choices=['angle', 'tan'])
    parser.add_argument("--method", default='num', choices=['num', 'num2', 'rank','rank2','skip','rank_binary','rank_binary2','rank_exp','num_exp','random'])
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
    parser.add_argument('--threshold', type=float, default=0.4, help='adaptive train threshold')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='adaptive train ratio')
    parser.add_argument('--train_epsilon', type=float, default=0.031, help='adaptive train ratio')
    parser.add_argument('--k', type=int, default=4, help='number of classes')
    args = vars(parser.parse_args())
    args['file_name'] = args['file_name'] + '_' + args['criterion'] + '_' + args['method']


    args['alpha'] = 2 / 255
    args['num_k'] = 7
    args['epsilon'] = 8 / 255
    args['batch_size'] = 100
    args['print_every'] = 250


    args['epsilon'] = 16/255
    args['alpha'] = args['epsilon']/4
    args['k'] = 16
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
