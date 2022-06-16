import os
import torch
import argparse
import numpy as np
from setup.utils import loaddata, loadmodel, savefile
from sklearn.neighbors import KNeighborsClassifier
import pickle
import platform
from tqdm import tqdm


def load_CIFAR101(train_loader):
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
    X_train, y_train = load_CIFAR101(train_loader)
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    '''
    X_train, y_train, X_test, y_test = load_CIFAR10(args['root_cifar'])

    # Checking the size of the training and testing data
    print('Training data shape: ', X_train.shape) #(50000, 32, 32, 3)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print(y_test)
    a = np.arange(start=1, stop=50001, step=1)
    '''
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    #X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print('==> Training KNN starts..')
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # save the model to disk
    filename = './models/finalized_knn.sav'
    pickle.dump(knn, open(filename, 'wb'))

    # load the model from disk
    #knn = pickle.load(open(filename, 'rb'))
    #print(X_test[[0],:].shape)
    #print(knn.predict(X_test[[0],:]))
    #predict = knn.predict(X_train)
    #print(predict) # [47189 42769 21299 ... 13253 17940 29497]
    #print(y_train)
    #print(knn.predict(X_train))

    #np.save('./models/knn_X_test.npy', predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("-m", '--model', choices=["vgg16", "wrn"], default="wrn")
    parser.add_argument("--round", action="store_true", default=False, help='if true, round epsilon vector')
    parser.add_argument("--precision", type=int, default=4, help='precision of rounding the epsilon vector')
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    parser.add_argument("--root_cifar", default=r'/data/cifar-10-batches-py',help='the dir that contains the data folder')
    parser.add_argument("--result_dir", default=r'/data/tangent', help='the working directory that contains AA, AAA')
    parser.add_argument("--clean", action="store_true", default=False, help='if true, clean training')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false", default=True,
                        help="shuffle in training or not")
    parser.add_argument('--depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width', type=int, default=10, help='WRN width factor')

    args = vars(parser.parse_args())

    args['alpha'] = 0.01
    args['num_k'] = 7
    args['epsilon'] = 8 / 255
    args['batch_size'] = 100
    args['print_every'] = 250
    print(args)
    main(args)
