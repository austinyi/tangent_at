import os
import torch
import numpy as np
from sklearn.decomposition import PCA


def noise_along(encoded, m, i):
    encoded[:,i] = encoded[:,i] + torch.rand(m).cuda()
    return encoded

def compute_tangent_component(PC_dmn, x, x_adv):
    m = x.shape[0]
    x = x.reshape(m,-1)
    x_adv = x_adv.reshape(m,-1)
    comp = []
    for i in range(m):
        A = PC_dmn[:,i,:]
        AA = np.matmul(A.T, np.linalg.pinv(np.matmul(A, A.T)))
        b = np.matmul(x[i,:]-x_adv[i,:], AA) # (x-x_adv)*A(A^TA)^-1
        comp.append(np.linalg.norm(b))
    return np.array(comp)


def save_AA(args, autoencoder, x, result_dir, idx, k=10):
    encoded, _ = autoencoder(x)
    encoded = encoded.detach()
    m, d = encoded.shape

    PC_dmn = []
    for i in range(d):
        X = []
        for j in range(k):
            encoded_new = noise_along(encoded, m, i)
            x_rec = autoencoder.decode(x.shape[0], encoded_new)
            x_rec = x_rec.view(x_rec.shape[0],-1).detach().cpu().numpy()
            X.append(x_rec)
        X = np.stack(X)
        PCs = []
        for q in range(m):
            X_kn = X[:,q,:]
            pc1 = PCA(n_components=1).fit(X_kn).components_
            PCs.append(pc1)
        PCs = np.concatenate(PCs,axis=0)
        PC_dmn.append(PCs)
    PC_dmn = np.stack(PC_dmn)
    
    for i in range(PC_dmn.shape[1]):
        A = PC_dmn[:,i,:]
        np.save(os.path.join(result_dir,'A',args['dataset'],'A_'+str(idx[i].item())+'.npy'), A) # d * n
        AA = np.matmul(A.T, np.linalg.pinv(np.matmul(A, A.T)))
        np.save(os.path.join(result_dir,'AA',args['dataset'],'AA_'+str(idx[i].item())+'.npy'), AA) # n * d
    return


def save_AAA(args, autoencoder, x, result_dir, idx, k=10):
    encoded, _ = autoencoder(x)
    encoded = encoded.detach()
    m, d = encoded.shape

    PC_dmn = []
    for i in range(d):
        X = []
        for j in range(k):
            encoded_new = noise_along(encoded, m, i)
            x_rec = autoencoder.decode(x.shape[0], encoded_new)
            x_rec = x_rec.view(x_rec.shape[0],-1).detach().cpu().numpy()
            X.append(x_rec)
        X = np.stack(X)
        PCs = []
        for q in range(m):
            X_kn = X[:,q,:]
            pc1 = PCA(n_components=1).fit(X_kn).components_
            PCs.append(pc1)
        PCs = np.concatenate(PCs,axis=0)
        PC_dmn.append(PCs)
    PC_dmn = np.stack(PC_dmn)
    
    for i in range(PC_dmn.shape[1]):
        A = PC_dmn[:,i,:]
        AAA = np.matmul(np.matmul(A.T, np.linalg.pinv(np.matmul(A, A.T))), A) #A(A^TA)^-1A^T
        np.save(os.path.join(result_dir,'AAA',args['dataset'],'AAA_'+str(idx[i].item())+'.npy'), AAA)
    return


def check_tangent(autoencoder, x, x_adv, k=10):
    encoded, _ = autoencoder(x)
    encoded = encoded.detach()
    m, d = encoded.shape

    PC_dmn = []
    for i in range(d):
        X = []
        for j in range(k):
            encoded_new = noise_along(encoded, m, i)
            x_rec = autoencoder.decode(x.shape[0], encoded_new)
            x_rec = x_rec.view(x_rec.shape[0],-1).detach().cpu().numpy()
            X.append(x_rec)
        X = np.stack(X)
        PCs = []
        for q in range(m):
            X_kn = X[:,q,:]
            pc1 = PCA(n_components=1).fit(X_kn).components_
            PCs.append(pc1)
        PCs = np.concatenate(PCs,axis=0)
        PC_dmn.append(PCs)
    PC_dmn = np.stack(PC_dmn)
    components = compute_tangent_component(PC_dmn, x.detach().cpu().numpy(), x_adv.detach().cpu().numpy())
    return PC_dmn, components


def compute_tangent(args, result_dir, idx, x, x_adv):
    m = x.shape[0]
    x = x.reshape(m,-1)
    x_adv = x_adv.reshape(m,-1)
    comp = []
    for i in range(len(idx)):
        #AAA = torch.tensor(np.load(os.path.join(result_dir,'AAA',args['dataset'],'AAA_'+str(idx[i].item())+'.npy'))).cuda() #A(A^TA)^-1A^T, n by n
        #AA = torch.tensor(np.load(os.path.join(result_dir,'AA',args['dataset'],'AA_'+str(idx[i].item())+'.npy'))).cuda()
        #b = torch.matmul(x_adv[i,:]-x[i,:], AA) # (x_adv-x)*A(A^TA)^-1
        AA = torch.tensor(np.load(os.path.join(result_dir,'AA',args['dataset'],'AA_'+str(idx[i].item())+'.npy'))).cuda() #AA: A(A^TA)^-1, n by d
        A = torch.tensor(np.load(os.path.join(result_dir,'A',args['dataset'],'A_'+str(idx[i].item())+'.npy'))).cuda() #A: A^T, d by n
        AAA = torch.matmul(AA, A) # A(A^TA)^-1A^T
        b = torch.matmul(AAA, x_adv[i,:]-x[i,:]) # A(A^TA)^-1A^T*(x_adv-x), n by 1
        comp.append(torch.norm(b).item())
    return np.array(comp)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / torch.norm(vector)


#<w,x^a-x>/(||w||*||x'-x||) = \sqrt{<w,x^a-x>/(||x'-x||*||x'-x||)}
#= \sqrt{<(A^TA)^{-1}A^T(x^a-x),A^T(x^a-x)>/(||x'-x||*||x'-x||)}

def compute_angle(args, result_dir, idx, x, x_adv):
    m = x.shape[0]
    x = x.reshape(m,-1)
    x_adv = x_adv.reshape(m,-1)
    Angles = []
    for i in range(len(idx)):
        z = x_adv[i,:]-x[i,:] # n by 1
        AA = torch.tensor(np.load(os.path.join(result_dir,'AA',args['dataset'],'AA_'+str(idx[i].item())+'.npy'))).cuda() #AA: A(A^TA)^-1, n by d
        nominator_part1 = torch.matmul(AA.T,z) # (A^TA)^{-1}A^T(x^a-x) --> d * 1
        A = torch.tensor(np.load(os.path.join(result_dir,'A',args['dataset'],'A_'+str(idx[i].item())+'.npy'))).cuda() #A: A^T, d by n
        nominator_part2 = torch.matmul(A,z) # A^T(x^a-x) --> d * 1
        nominator = torch.matmul(nominator_part1.T, nominator_part2)
        denom = torch.matmul(z.T,z)
        value = torch.sqrt(nominator/denom)
        angle = np.arccos(np.clip(value.item(),-1,1))
        Angles.append(angle)
        # Previous method
        #AAA = torch.tensor(np.load(os.path.join(result_dir,'AAA',args['dataset'],'AAA_'+str(idx[i].item())+'.npy'))).cuda() #A(A^TA)^-1A^T
        #w = torch.matmul(AAA, z) # A(A^TA)^-1A^T*(x_adv-x)
        #z = unit_vector(z)
        #w = unit_vector(w)
        #angle = np.arccos(np.clip(torch.dot(w,z).item(),-1,1))
        #Angles.append(angle)
    return np.array(Angles)