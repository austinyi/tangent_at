import os
#os.chdir(r'D:\yaoli\tangent')
import torch
import argparse
from tqdm import tqdm
from train_ae import Autoencoder
from tangent import save_AA, save_AAA
from setup.utils import loaddata
from setup.setup_pgd import to_var


def make_directories(args, result_dir):
    A_dir = os.path.join(result_dir,'A',args['dataset'])
    AA_dir = os.path.join(result_dir,'AA',args['dataset'])
    AAA_dir = os.path.join(result_dir,'AAA',args['dataset'])
    os.makedirs(A_dir, exist_ok=True)
    os.makedirs(AA_dir, exist_ok=True)
    os.makedirs(AAA_dir, exist_ok=True)    
    return

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


def saveA_AA_AAA(args, autoencoder, train_loader, result_dir):
    for idx, x, target in tqdm(train_loader):            
        x, target = to_var(x), to_var(target)
        save_AA(args, autoencoder, x, result_dir, idx, k=args['k'])
        #save_AAA(args, autoencoder, x, result_dir, idx, k=10)
    return


def main(args):
    result_dir = args['result_dir']
    make_directories(args, result_dir)
    print('==> Loading data..')
    train_loader, _ = loaddata(args)
    
    print('==> Loading model..')
    autoencoder = load_autoencoder(args)
    
    print('==> Generating components..')
    saveA_AA_AAA(args, autoencoder, train_loader, result_dir)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("--init", default='cifar10_plain')
    parser.add_argument("--root", default='/proj/STOR/yaoli', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default='/proj/STOR/yaoli', help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default='/pine/scr/y/a/yaoli/data', help='the working directory that contains AA, AAA')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--ae_load", default='ae_loss0.589.pt',
                        help="name of the autoencoder to load.")
    parser.add_argument("--train_shuffle", action="store_true",  default=False,
                        help="do not shuffle here since we're computing the A over training set in order")    
    args = vars(parser.parse_args())
    if args['dataset'] == 'mnist':
        args['batch_size'] = 100
    elif args['dataset'] == 'cifar10':
        args['batch_size'] = 100
        args['dim'] = 128
        args['k'] = 10
    elif args['dataset'] == 'stl10':
        args['batch_size'] = 64
    elif args['dataset'] == 'tiny':
        args['batch_size'] = 128
        args['num_gpu'] = 2
    else:
        print('invalid dataset')
    print(args)
    main(args)