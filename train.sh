python resnet.py -a alexnet  --dist-url 'tcp://localhost:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /fs/vulcan-datasets/imagenet

python cifar10.py --model VGG16
python cifar10.py --model VGG16_NoPooling