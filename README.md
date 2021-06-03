1) To generate the experiments for Figure 1 of the paper, run  <code>python simple.py</code>.
2) To train models on MNIST use <code>python train_mnist_full.py</code> 
and specify the parameters for network architecture and 
adversarial attack to use. Use the --fashion-mnist flag
to train FashionMNIST dataset.
3) To train models on SVHN dataset use `python train_svhn.py`,
 with required parameters. To evaluate robustness of trained models,
 use `python eval_svhn.py`
4) For results on Table 1 and Table 2, use orthogonal_vectors.ipynb to
 train on orthogonal vectors and Frequency Adversarial.ipynb to train on orthogonal frequencies.
 
5) Cifar10 models are generated with code from https://github.com/kuangliu/pytorch-cifar.
 And ImageNet models are generated using pytorch model zoo. To test performance,
 on CIFAR-10 use `python test_cifar.py`  and for Imagenet use `python test_imagenet.py` and specify the
 model architecture using `--model` argument. 
6) To test robustness of Vision Transformers use `python test_vit_imagenet.py`
7) To train a Sparse MLP on CIFAR-10 following [1], use `python train_cifar_sparse_mlp.py`. There's
    a few differences between our training and original paper, we use default data augmentations instead of 
    fastautoaugment as proposed in the paper and train for 2000 epochs instead of 4000. 
    We trained with different lambda values as proposed in the paper for our results.
 
### References
[1] Towards Learning Convolutions from Scratch (https://arxiv.org/abs/2007.13657)
