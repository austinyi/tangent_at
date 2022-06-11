import os
#os.chdir(r'D:\yaoli\tangent')
import torch
import argparse
from setup.utils import loaddata, loadmodel
from setup.setup_pgd_adaptive import LinfPGDAttack, attack_over_test_data, GA_PGD

def testClassifier(test_loader, model, use_cuda=True, batch_size=100):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    print("The prediction accuracy on testset is {}".format(acc))
    return acc


def testattack(args, classifier, test_loader, use_cuda=True):
    classifier.eval()
    adversary = LinfPGDAttack(classifier, epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
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
    _, test_loader = loaddata(args)
    
    model_list = args['file_list']
    
    for model_name in model_list:
        print('==> Loading model {}'.format(model_name))
        args['init'] = model_name
        model = loadmodel(args)
        
        if use_cuda:
            model = model.cuda()
        print('==> Evaluating {}'.format(model_name))
    
        testClassifier(test_loader,model,use_cuda=use_cuda,batch_size=args['batch_size'])
        testattack(args, model, test_loader, use_cuda=use_cuda)
        test_pgd20_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4, loss_fn="cent",
                category="Madry", random=True)
        print(test_pgd20_acc)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("--file_list", default=['cifar10_42adv_angle0.814','cifar10_425adv_angle0.82','cifar10_43adv_angle0.82'], help='list of models to test')
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width', type=int, default=10, help='WRN width factor')
    parser.add_argument("--init", default=None)
    parser.add_argument("--root", default='/data', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--root_data", default='/', help='D:\yaoli, /proj/STOR/yaoli')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false",  default=True,
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