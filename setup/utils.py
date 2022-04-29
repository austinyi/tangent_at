import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from setup.setup_pgd import LinfPGDAttack, attack_over_test_data
from setup.setup_data import load_filenames_labels, ValData, TrainData
from setup.setup_loader import CIFAR10


class MNIST:
    def __init__(self, root):
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root=os.path.join(root,'data'), train=True, transform=trans, download=False)
        test_set = datasets.MNIST(root=os.path.join(root,'data'), train=False, transform=trans, download=False)
        
        self.train_data = train_set
        self.test_data = test_set   
    
    
def loaddata(args):
    if args['dataset'] == 'mnist':
        train_loader = DataLoader(MNIST(args['root_data']).train_data, batch_size=args['batch_size'], shuffle=args['train_shuffle'])
        test_loader = DataLoader(MNIST(args['root_data']).test_data, batch_size=args['batch_size'], shuffle=False)
    elif args['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        trainset = CIFAR10(root=os.path.join(args['root_data'],'data'),
                                 train=True,download=False,transform=transform_train) #return index as well
        # trainset = datasets.CIFAR10(root=os.path.join(args['root_data'],'data'),
        #                         train=True,download=False,transform=transform_train)        
        train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['train_shuffle'])                
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=os.path.join(args['root_data'],'data'),
                                train=False,download=False,transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)  
    elif args['dataset'] == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.STL10(root=os.path.join(args['root_data'],'data'),
                                 split='train',download=False,transform=transform_train)
        testset = datasets.STL10(root=os.path.join(args['root_data'],'data'),
                                split='test',download=False,transform=transform_test)
        train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['train_shuffle'])
        test_loader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
    elif args['dataset'] == 'tiny':
        labels_train = load_filenames_labels('train', args['root_data'])    
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(0.4, 0.4, 0.4),
                                              transforms.ToTensor()])
        train_dataset = TrainData(range(100000), labels_train, transform_train)
        train_loader =  DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=args['train_shuffle'],num_workers=args['num_gpu']) 
        labels_val = load_filenames_labels('val', args['root_data'])
        transform_test = transforms.Compose([transforms.ToTensor()])
        test_dataset = ValData(range(10000), labels_val, transform_test)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,num_workers=args['num_gpu'])
    else:
        print("unknown dataset")
    return train_loader, test_loader


def loadmodel(args):
    if args['dataset'] == 'mnist':
        from setup.setup_mnist import BasicCNN
        model = BasicCNN()           
    elif args['dataset'] == 'cifar10':       
        from setup.setup_vgg import vgg16
        model = vgg16()
    elif args['dataset'] == 'stl10':       
        from setup.setup_svhn import SVHN32   
        model = SVHN32()
    elif args['dataset'] == 'tiny':       
        from setup.setup_tiny import vgg16_new  
        model = nn.DataParallel(vgg16_new())
    else:
        print("unknown model")
        return
    if args['init'] != None:
        print("Loading model init {}".format(args['init']))
        model.load_state_dict(torch.load(os.path.join('./models/'+args['dataset'], args['init'])))
    return model
    
       
def testattack(classifier, test_loader, epsilon, k, alpha, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=epsilon, k=k, a=alpha)
    param = {
    'test_batch_size': 100,
    'epsilon': epsilon,
    }            
    acc = attack_over_test_data(classifier, adversary, param, test_loader, use_cuda=use_cuda)
    return acc

def savefile(file_name, model, dataset):
    if file_name != None:
        root = "./models/"+dataset
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(model.state_dict(), os.path.join(root,file_name))
    return


def initial_no_back_track(file_name, state_dict, model):
    import collections
    new_keys = list(model.state_dict().keys())
    keys = list(state_dict.keys())
    new_state_dict = collections.OrderedDict()
    count = 0
    for i in range(len(new_keys)):
        j = i - count
        key_name = new_keys[i]
        if 'num_batches_tracked' in key_name:
            count += 1
            new_state_dict[new_keys[i]] = model.state_dict()[new_keys[i]]
            continue
        new_state_dict[new_keys[i]] = state_dict[keys[j]]
    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(),file_name)  
    return  







