import os
import torch
import argparse
import numpy as np
from setup.utils import loaddata, loadmodel, savefile
from sklearn.neighbors import KNeighborsClassifier
import pickle
import platform
from tqdm import tqdm


def load_CIFAR10(train_loader):
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
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)
    X_train, y_train = load_CIFAR10(train_loader)
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
    #knn.fit(X_train, y_train)

    # save the model to disk
    filename = './models/finalized_knn.sav'
    #pickle.dump(knn, open(filename, 'wb'))


    # load the model from disk
    knn = pickle.load(open(filename, 'rb'))
    #print(X_test[[0],:].shape)
    #print(knn.predict(X_test[[0],:]))
    predict = knn.predict(X_train)
    print(predict) # [47189 42769 21299 ... 13253 17940 29497]
    print(X_train[0])
    #print(y_train)
    #print(knn.predict(X_train))

    #np.save('./models/knn_X_test.npy', predict)
    pbar = tqdm(train_loader)
    for idx, X, y in pbar:
        print(X[0])
        print(idx)
        X_knn = X.cpu().numpy()
        X_knn = np.reshape(X_knn, (X_knn.shape[0], -1))
        predict_idx1 = knn.predict(X_knn)
        print(predict_idx1)
        '''
        y_pred_adv = pred_batch(X_adv, classifier)
        corr_idx = y_pred_adv.numpy() == y.numpy()

        angle = compute_angle(args, args['result_dir'], predict_idx, X_train[predict_idx], X_adv)
        tangent = compute_tangent(args, args['result_dir'], predict_idx, X_train[predict_idx], X_adv)

        correct_angle = np.append(correct_angle, angle[corr_idx])
        wrong_angle = np.append(wrong_angle, angle[np.invert(corr_idx)])
        correct_tangent = np.append(correct_tangent, tangent[corr_idx])
        wrong_tangent = np.append(wrong_tangent, tangent[np.invert(corr_idx)])
        '''
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10", "stl10", "tiny"], default="cifar10")
    parser.add_argument("--init", default=None, help='initial the model with pre-trained one')
    parser.add_argument("--root", default=r'/data', help='the directory that contains the project folder')
    parser.add_argument("--root_data", default=r'/', help='the dir that contains the data folder')
    #parser.add_argument("--root_cifar", default=r'/data/cifar-10-batches-py',help='the dir that contains the data folder')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--train_shuffle", action="store_false", default=False,
                        help="shuffle in training or not")

    args = vars(parser.parse_args())

    args['alpha'] = 0.01
    args['num_k'] = 7
    args['epsilon'] = 8 / 255
    args['batch_size'] = 100
    args['print_every'] = 250
    print(args)
    main(args)
