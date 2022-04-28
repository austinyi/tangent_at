import os
#os.chdir(r'D:\yaoli\tangent')
import math
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from tangent import check_tangent, save_AA, save_AAA
from train_ae import Autoencoder
from setup.utils import loaddata, loadmodel
from scipy.interpolate import make_interp_spline
from setup.setup_pgd import to_var, adv_train, pred_batch, LinfPGDAttack

# Empirical study of tangent components range


def load_autoencoder(args):
    autoencoder = Autoencoder(args['dim'])
    if args['ae_load']:
        print("Loading pre-trained models {}".format(args['ae_load']))
        state_dict = torch.load(os.path.join(args['model_folder'],args['ae_load']), map_location=torch.device('cpu'))
        autoencoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def compute_tangent_components(args, model, autoencoder, train_loader, result_dir):
    # model = model.cuda()
    # adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    # criterion = nn.CrossEntropyLoss()
    # X_ori = []
    # X_advs = []
    # X_components = []
    # labels = []
    for idx, x, target in tqdm(train_loader):            
        x, target = to_var(x), to_var(target)
        # target_pred = pred_batch(x, model)
        # x_adv = adv_train(x, target_pred, model, criterion, adversary)
        # save_AA(args, autoencoder, x, result_dir, idx, k=10)
        save_AAA(args, autoencoder, x, result_dir, idx, k=10)
        # components = check_tangent(autoencoder, x, x_adv, idx, k=10)        
        # X_advs.append(x_adv.detach().cpu().numpy())
        # X_components.append(components)
        # labels.append(target.detach().cpu().numpy())
        # X_ori.append(x.detach().cpu().numpy())

    # X_advs = np.concatenate(X_advs, axis=0)
    # X_components = np.concatenate(X_components)
    # labels = np.concatenate(labels)
    # X_ori = np.concatenate(X_ori, axis=0)

    # print("Components Stat:")
    # print("Max: {:.3f}".format(X_components.max()))
    # print("Min: {:.3f}".format(X_components.min()))
    # print("Mean: {:.3f}".format(X_components.mean()))
    
    # np.save(os.path.join(result_dir, 'X_adv_'+args['dataset']+'.npy'), X_advs)
    # np.save(os.path.join(result_dir, 'comp_'+args['dataset']+'.npy'), X_components)
    # np.save(os.path.join(result_dir, 'labels_'+args['dataset']+'.npy'), labels)
    # np.save(os.path.join(result_dir, 'X_ori_'+args['dataset']+'.npy'), X_ori)
    return


def testClassifier(X_adv, labels, model, batch_size=100, verbose=False):
    model = model.cuda()
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    n = X_adv.shape[0]
    for i in range(math.floor(n/batch_size)):
        batch = X_adv[i*batch_size:(i+1)*batch_size]
        target = labels[i*batch_size:(i+1)*batch_size]
        with torch.no_grad(): 
            batch = torch.tensor(batch).cuda()
            target = torch.tensor(target).cuda()
            out = model(batch)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += batch.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    if verbose:
        print("The prediction accuracy on data is {:.4f}".format(acc))
    return acc


def one_step_update(args, model, batch, target, criterion, X_ori_test, labels):
    model_cp = copy.deepcopy(model)
    model_cp = model_cp.cuda()
    optimizer = torch.optim.SGD(model_cp.parameters(),lr=args['lr'],momentum=0.9, weight_decay=args['weight_decay'])  
    loss = criterion(model_cp(batch),target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    acc = testClassifier(X_ori_test, labels, model_cp, batch_size=100)
    return acc, loss.item()


def explore_adv_comp(args, model):
    batch_size = args['batch_size']
    result_dir = os.path.join(args['root'], 'tangent', 'results')
    X_ori_test = np.load(os.path.join(result_dir, 'X_ori_test_'+args['dataset']+'.npy'))
    X_ori = np.load(os.path.join(result_dir, 'X_ori_'+args['dataset']+'.npy'))
    X_adv = np.load(os.path.join(result_dir, 'X_adv_'+args['dataset']+'.npy'))
    labels = np.load(os.path.join(result_dir, 'labels_'+args['dataset']+'.npy'))
    labels_test = np.load(os.path.join(result_dir, 'labels_test_'+args['dataset']+'.npy'))
    components = np.load(os.path.join(result_dir, 'comp_'+args['dataset']+'.npy'))
    
    # Rank the images based on components values
    ind = np.argsort(components)
    components = components[ind]
    X_ori = X_ori[ind]
    X_adv = X_adv[ind]
    labels = labels[ind]
    
    testClassifier(X_ori, labels, model, verbose=True)
    testClassifier(X_adv, labels, model, verbose=True)
    
    # Get the change of clean accuracy and robust accuracy after adversarial training for each batch
    criterion = nn.CrossEntropyLoss()
    n = X_adv.shape[0]
    losses = []
    accuracies = []
    avg_comp = []
    for i in tqdm(range(math.floor(n/batch_size))):
        batch = X_adv[i*batch_size:(i+1)*batch_size]
        target = labels[i*batch_size:(i+1)*batch_size]
        batch = torch.tensor(batch).cuda()
        target = torch.tensor(target).cuda()
        acc, loss = one_step_update(args, model, batch, target, criterion, X_ori_test, labels_test)
        losses.append(loss)
        accuracies.append(acc)
        avg_comp.append(np.mean(components[i*batch_size:(i+1)*batch_size]))
    
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    avg_comp = np.array(avg_comp)
    np.save(os.path.join(result_dir, 'losses_'+args['dataset']+'.npy'), losses)
    np.save(os.path.join(result_dir, 'accuracies_'+args['dataset']+'.npy'), accuracies)
    np.save(os.path.join(result_dir, 'avg_comp_'+args['dataset']+'.npy'), avg_comp)
    # Expect to see high component --> high test accuracy; high component --> low loss
    # Draw plot to see the trend
    
    # Get smoothing curves
    xnew = np.linspace(avg_comp.min(), avg_comp.max(), 200) 
    spl1 = make_interp_spline(avg_comp, accuracies, k=3)
    y_smooth1 = spl1(xnew)
    
    spl2 = make_interp_spline(avg_comp, losses, k=3)
    y_smooth2 = spl2(xnew)    
    
    f = plt.figure(figsize=(4.8,3.6))
    plt.scatter(avg_comp, accuracies)
    plt.plot(xnew, y_smooth1,'--', label="Accuracy", color='cyan')
    plt.legend()
    plt.title("CIFAR10", fontsize=16)
    plt.xlabel("Tangent Components", fontsize=14)
    plt.ylabel("Testing accuracy", fontsize=14)
    plt.show()
    f.savefig("./pics/cifar01.pdf", bbox_inches='tight')    
    
    f = plt.figure(figsize=(4.8,3.6))
    plt.scatter(avg_comp, losses)
    plt.plot(xnew, y_smooth2,'--',label="Loss", color='cyan')
    plt.legend()
    plt.title("CIFAR10", fontsize=16)
    plt.xlabel("Tangent Components", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()
    f.savefig("./pics/cifar02.pdf", bbox_inches='tight')    
    return

def main(args):
    result_dir = os.path.join(args['root'], 'tangent', 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    print('==> Loading data..')
    train_loader, _ = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)
    autoencoder = load_autoencoder(args)
    
    print('==> Generating components..')
    compute_tangent_components(args, model, autoencoder, train_loader, result_dir)
    
    # print('==> Analyzing the relationship between components and clean accuracy..')
    # explore_adv_comp(args, model)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("--init", default='cifar10_plain')
    parser.add_argument("--root", default='/proj/STOR/yaoli', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--root_data", default='/proj/STOR/yaoli', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--ae_load", default='ae_loss0.589.pt',
                        help="name of the autoencoder to load.")
    parser.add_argument("--train_shuffle", action="store_true",  default=False,
                        help="shuffle in training or not")    
    args = vars(parser.parse_args())
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.01
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['batch_size'] = 100
        args['dim'] = 128
        args['lr'] = 1e-3
        args['weight_decay'] = 1e-4
    elif args['dataset'] == 'stl10':
        args['alpha'] = 0.0156
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['batch_size'] = 64
    elif args['dataset'] == 'tiny':
        args['alpha'] = 0.002
        args['num_k'] = 10
        args['epsilon'] = 0.01
        args['batch_size'] = 128
        args['num_gpu'] = 2
    else:
        print('invalid dataset')
    print(args)
    main(args)