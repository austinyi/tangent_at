import os
#os.chdir(r'D:\yaoli\tangent')
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from train_ae import Autoencoder
from tangent import check_tangent
from setup.utils import loaddata, loadmodel, savefile
from setup.setup_pgd import to_var, adv_train, pred_batch, LinfPGDAttack, attack_over_test_data


def select_adv(components, thrds, criterion='upper'):
    if criterion == 'upper':
        use = (components > thrds[1])
    elif criterion == 'middle' and len(thrds)==2:
        use = np.logical_and(components >= thrds[0], components <= thrds[1])
    elif criterion == 'lower':
        use = (components < thrds[0])
    else:
        raise Exception("No such criterion")
    return use


def load_autoencoder(args):
    autoencoder = Autoencoder(args['dim'])
    if args['ae_load']:
        print("Loading pre-trained models {}".format(args['ae_load']))
        state_dict = torch.load(os.path.join(args['model_folder'],args['ae_load']))
        autoencoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder
   
def trainClassifier(args, model, autoencoder, train_loader, test_loader, use_cuda=True):    
    if use_cuda:
        model = model.cuda()
        
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])  
    train_criterion = nn.CrossEntropyLoss()
    for epoch in range(args['num_epoch']):
        # trainning
        ave_loss = 0
        step = 0
        for x, target in tqdm(train_loader):            
            x, target = to_var(x), to_var(target)
            if args['clean']:
                x_adv_new = x
            else:
                target_pred = pred_batch(x, model)
                x_adv = adv_train(x, target_pred, model, train_criterion, adversary)
                components = check_tangent(autoencoder, x, x_adv, k=args['comp_k'])
                use = select_adv(components, args['thrd'], args['criterion'])
                x_adv_new = x_adv[use,:,:,:]
                target = target[use]
                if x_adv_new.shape[0] < 2:
                    continue
            loss = train_criterion(model(x_adv_new),target)
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
    autoencoder = load_autoencoder(args)
    
    print('==> Training starts..')            
    model = trainClassifier(args, model, autoencoder, train_loader, test_loader, use_cuda=use_cuda) 
    testClassifier(test_loader,model,use_cuda=use_cuda,batch_size=args['batch_size'])
    testattack(model, test_loader, args, use_cuda=use_cuda)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("-n", "--num_epoch", type=int, default=30)
    parser.add_argument("-f", "--file_name", default="cifar10_adv")
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("--criterion", default="upper", choices=["upper", "middle","lower"])
    parser.add_argument("--init", default='cifar10_plain')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--root", default='/proj/STOR/yaoli')
    parser.add_argument("--root_data", default='/proj/STOR/yaoli', help='D:\yaoli')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--ae_load", default='ae_loss0.589.pt',
                        help="name of the autoencoder to load.")
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
        args['dim'] =128
        args['thrd'] = [0.39, 0.43]
        args['comp_k'] = 10
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