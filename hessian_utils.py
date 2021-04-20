import torch
from torch import nn

def normalize(X):
    # mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
    # std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).cuda()
    mu = 0
    std = 1
    return (X - mu)/std


def orthnormal(v, eigenvectors):
    '''
    Takes a matrix of random vectors (batch x input_size) and a list of
    eigenvectors with each element containing eigenvector for entire batch
    (batch x input_size) and returns the normalized random vectors orthonormal
    to all the eigenvectors in the list.
    '''
    for eigenvector in eigenvectors:
        alpha = (v * eigenvector).sum(dim=1)
        v -= alpha.unsqueeze(1)*eigenvector
    return torch.nn.functional.normalize(v)


def tolerance(eigenvalues, new_eigenvalues, tolerance):
    return np.all((abs(eigenvalues - new_eigenvalues) < tolerance).cpu().numpy())


class Hessian():
        """
        The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
        """
        def __init__(self, model, criterion=nn.CrossEntropyLoss(reduction='sum'),  cuda=True):
            # make sure model is in evaluation model,
            # don't need gradients with parameters.
            self.model = model.eval()  
            self.criterion = criterion  
            if cuda:
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        def eigenvalues(self, data, maxIter=100, tol=1e-3, top_n=1):
            """
            compute the top_n eigenvalues using power iteration method
            maxIter: maximum iterations used to compute each single eigenvalue
            tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
            top_n: top top_n eigenvalues will be computed
            """
            assert top_n >= 1
            inputs, targets = data
            if self.device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            # flattening to generate a matrix of batches with eigenvalues for each
            inputs_flat = inputs.view(inputs.shape[0], -1)
            # get gradient with input (flattening for ease of representation)
            inputs_flat.requires_grad_()
            inputs = inputs_flat.view(inputs.shape)
            logits = self.model(normalize(inputs))
            loss = self.criterion(logits, targets)
            # calculate input gradient and create graph for higher order derivatives.
            inputs_grad = torch.autograd.grad(loss, inputs_flat, create_graph=True)[0]
            device = self.device
            eigenvalues = []
            eigenvectors = []
            computed_dim = 0
            while computed_dim < top_n:
                eigenvalue = None
                v = torch.randn_like(inputs_flat)  # generate random vector like inputs
                if self.device=='cuda':
                    v = v.cuda()
                v = torch.nn.functional.normalize(v)  # normalize the vector
                for i in range(maxIter):
                    # project to orthogonal subspace to currently computed eigenvectors
                    # like Graham-Schmidt process.
                    v = orthnormal(v, eigenvectors)
                    # Get the hessian vector product using torch autograd.
                    # can also try using the functional hvp implementation of torch autograd
                    # in the new version of pytorch.
                    Hv = torch.autograd.grad(inputs_grad.mul(v).sum(), inputs_flat, retain_graph=True)[0]
                    # v.T H v for eigenvalue computation
                    tmp_eigenvalue = (v * Hv).sum(1, keepdim=True)
                    # re-normalize the vector
                    v = torch.nn.functional.normalize(Hv)
                    if eigenvalue == None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if tolerance(eigenvalue, tmp_eigenvalue, tol):
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                # vector after max-iterations turns into eigenvector
                # project to orth subspace.
                v = orthnormal(v, eigenvectors)
                eigenvalues.append(eigenvalue)
                eigenvectors.append(v)
                computed_dim += 1
            return eigenvalues, eigenvectors, inputs_grad.clone()