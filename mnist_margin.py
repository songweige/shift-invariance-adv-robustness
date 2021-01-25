import torchvision
import torchvision.transforms as transforms

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(
    root='/vulcanscratch/songweig/datasets/mnist', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='/vulcanscratch/songweig/datasets/mnist', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)



dc_components = {i:[] for i in range(10)}
for batch_idx, (inputs, targets) in enumerate(trainloader):
	if len(dc_components[targets[0].item()])>= 20:
		continue
	dc_component = inputs.sum()
	dc_components[targets[0].item()].append(dc_component.item())


for i in range(10):
	print('Class %d, number of sample: %d, minimal dc component: %.2f, maximal dc compoenent: %.2f'%(i, len(dc_components[i]), min(dc_components[i])/28., max(dc_components[i])/28.))





import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline

def load_data(folder, train=True, classes = [1,2], samples=100, shifted=False):
    train_file = 'training.pt'
    test_file = 'test.pt'
    data_file = train_file if train else test_file
    data, targets = torch.load(os.path.join(folder, data_file))
    select_indexes = [i for i, x in enumerate(targets.tolist()) if x in classes]
    data, targets = data[select_indexes], targets[select_indexes]
    targets[targets==classes[0]] = 0
    targets[targets==classes[1]] = 1
    targets = targets.float()
    if samples!=None:
        data = data[:samples,:,:]
        targets = targets[:samples]
    data, targets = data.numpy(), targets.numpy()
    if shifted:
        data_all = data
        targets_all = targets
        for x in range(28):
            for y in range(28):
                if x==28 and y==28:
                    continue
                shifted_data = np.roll(data, x, axis=1)
                shifted_data = np.roll(shifted_data, y, axis=2)
                data_all = np.concatenate([data_all, shifted_data])
                targets_all = np.concatenate([targets_all, targets])
        return data_all.reshape(data_all.shape[0], -1), targets_all
    return data.reshape(data.shape[0], -1), targets



classes = [0,1]
X,y = load_data('/vulcanscratch/songweig/datasets/mnist/MNIST/processed', classes = classes, samples=40, shifted=True)
X,y = load_data('/vulcanscratch/songweig/datasets/mnist/MNIST/processed', classes = classes, samples=40, shifted=False)

X_test, y_test = load_data('../datasets/MNIST/processed', samples=None, train=False, classes=classes)


X1 = X[y.astype(bool)]
X1_dc = clf[0].transform(X1).sum(-1)/28.
X1_dc.min(), X1_dc.max()
X2 = X[(1-y).astype(bool)]
X2_dc = clf[0].transform(X2).sum(-1)/28.
X2_dc.min(), X2_dc.max()


clf = make_pipeline(StandardScaler(), svm.LinearSVC(C=1000))
clf.fit(X, y)
clf.score(X, y)
clf.score(X_test, y_test)

w = clf[1].coef_[0]
b = clf[1].intercept_[0]
for x, y_true in zip(X, y):
	# assert(clf[1].predict(x[None, :])[0] == y_true)
	assert(np.logical_not(np.logical_xor(np.einsum('i,i', w, clf[0].transform(x[None, :])[0])+b>0, y_true>0)))

2/np.linalg.norm(w, ord=2)
clf[0].mean_