import os
os.chdir(r'D:\yaoli\tangent')
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from tangent import compute_angle, compute_tangent_torch
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd_adaptive import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data

PI = math.pi


def trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=True):    
    if use_cuda:
        model = model.cuda()
        
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])  
    train_criterion = nn.CrossEntropyLoss()
    for epoch in range(args['num_epoch']):
        # trainning
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
                    ep = (1/(angles*np.max(1/angles)))*args['epsilon']
                    x_adv = adv_train(x, target_pred, model, train_criterion, adversary, ep=ep)
                elif args['criterion'] == 'tan':
                    components = compute_tangent_torch(args, result_dir, idx, x, x_adv_init)
                    ep = components/np.max(components)*args['epsilon']
                    x_adv = adv_train(x, target_pred, model, train_criterion, adversary, ep=ep)
                else:
                    raise Exception("No such criterion")    
            loss = train_criterion(model(x_adv),target)
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1    
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


def testBatch(model, x, x_adv, target):
    model.eval()
    diff = x_adv - x
    print("The max diff is", diff.max())
    print("The min diff is", diff.min())
    names = ['clean', 'adv']
    j = 0
    for sample in [x, x_adv]:
        out = model(sample)
        _, pred_label = torch.max(out.data, 1)
        acc = (pred_label == target.data).sum().double()/sample.shape[0]
        print("{} accuracy is {:.3f}".format(names[j], acc.item()))
        j += 1
    for k in range(diff.shape[0]):
        print(diff[k].min(), diff[k].max())
    return acc


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


def main(args):
    use_cuda = torch.cuda.is_available()
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)
    
    print('==> Training starts..')
    if args['criterion'] == 'tan':      
        result_dir = os.path.join(args['root'], 'tangent', 'results')
    elif args['criterion'] == 'angle':
        result_dir = '/pine/scr/y/a/yaoli/data'
    model = trainClassifier(args, model, result_dir, train_loader, test_loader, use_cuda=use_cuda) 
    testClassifier(test_loader,model,use_cuda=use_cuda,batch_size=args['batch_size'])
    testattack(model, test_loader, args, use_cuda=use_cuda)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("-n", "--num_epoch", type=int, default=100)
    parser.add_argument("-f", "--file_name", default="cifar10_adapt")
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("--criterion", default='tan', choices=['angle','tan'])
    parser.add_argument("--init", default='cifar10_plain')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--root", default=r'D:\yaoli', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--root_data", default=r'D:\yaoli', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false",  default=True,
                        help="shuffle in training or not")    
    args = vars(parser.parse_args())
    args['file_name'] = args['file_name']+'_'+args['criterion']
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
        args['print_every'] = 300
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.01
        args['num_k'] = 20
        args['epsilon'] = 0.03
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