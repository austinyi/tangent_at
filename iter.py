# importing libraries
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm


# class to represent dataset
class HeartDataSet():

    def __init__(self):
        # loading the csv file from the folder path
        data1 = np.loadtxt('heart.csv', delimiter=',',
                           dtype=np.float32, skiprows=1)

        # here the 13th column is class label and rest
        # are features
        self.x = torch.from_numpy(data1[:, :13])
        self.y = torch.from_numpy(data1[:, [13]])
        self.n_samples = data1.shape[0]

    # support indexing such that dataset[i] can
    # be used to get i-th sample
    def __getitem__(self, index):
        return index, self.x[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = HeartDataSet()


# Loading whole dataset with DataLoader
# shuffle the data, which is good for training
dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

# total samples of data and number of iterations performed
total_samples = len(dataset)
n_iterations = total_samples//100

num_epochs = 4

print(next(dataloader))
print(next(dataloader))
print(next(dataloader))
print(next(dataloader))

print(n_iterations)
for epoch in range(num_epochs):
	for i, inputs, labels in tqdm(dataloader):

		# here: 303 samples, batch_size = 4, n_iters=303/4=75 iterations
		# Run our training process
		#if (i+1) % 5 == 0:
			print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}|\
				Inputs {inputs.shape} | Labels {labels.shape}')
