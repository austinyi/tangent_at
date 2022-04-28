import os
os.chdir(r'D:\yaoli\tangent')
import math
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from tangent import compute_angle
from setup.utils import loaddata, loadmodel
from setup.setup_pgd import to_var, adv_train, pred_batch, LinfPGDAttack

# Empirical study of tangent components range


def compute_angle_components(args, model, train_loader, result_dir):
    model = model.cuda()
    save_dir = os.path.join(args['root'], 'tangent', 'results')
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    criterion = nn.CrossEntropyLoss()
    Angles = []
    for idx, x, target in tqdm(train_loader):            
        x, target = to_var(x), to_var(target)
        target_pred = pred_batch(x, model)
        x_adv = adv_train(x, target_pred, model, criterion, adversary)
        angle = compute_angle(args, result_dir, idx, x, x_adv)
        Angles.append(angle)
    Angles = np.concatenate(Angles)
    np.save(os.path.join(save_dir, 'Angles_'+args['dataset']+'.npy'), Angles)
    return

def explore_angle_comp(args):
    save_dir = os.path.join(args['root'], 'tangent', 'results')
    Angles = np.load(os.path.join(save_dir, 'Angles_'+args['dataset']+'.npy'))
    degrees = Angles*180/math.pi
    print("The angle mean: {:.3f}".format(degrees.mean()))
    print("The angle max: {:.3f}".format(degrees.max()))
    print("The angle min: {:.3f}".format(degrees.min()))
    print("The angle median: {:.3f}".format(np.median(degrees)))
    # Get the histogram
    import seaborn as sns
    f = plt.figure(figsize=(4.8,3.6))
    hist = sns.histplot(data=degrees)
    hist.annotate(str(round(degrees.min())),xy = (degrees.min(),0))
    hist.annotate(str(round(degrees.mean())),xy = (degrees.mean(),0))
    hist.annotate(str(round(degrees.max())),xy = (degrees.max(),0))
    f.savefig("./pics/cifar_angle.pdf", bbox_inches='tight')    
    return

def main(args):
    result_dir = '/pine/scr/y/a/yaoli/data'
    
    print('==> Loading data..')
    train_loader, _ = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)
    
    print('==> Generating components..')
    compute_angle_components(args, model, train_loader, result_dir)
    
    # print('==> Analyzing the relationship between components and clean accuracy..')
    explore_angle_comp(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("--init", default='cifar10_plain')
    parser.add_argument("--root", default=r'D:\yaoli', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--root_data", default=r'D:\yaoli', help='D:\yaoli, /proj/STOR/yaoli')
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