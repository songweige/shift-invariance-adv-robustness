python cifar10.py --model resnet_nobatchnorm_18
python test_cifar.py --model resnet_nobatchnorm_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_nobatchnorm_18.txt
python cifar10.py --model resnet_nobatchnorm_34
python test_cifar.py --model resnet_nobatchnorm_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_nobatchnorm_34.txt
python cifar10.py --model resnet_nobatchnorm_50
python test_cifar.py --model resnet_nobatchnorm_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_nobatchnorm_50.txt

python cifar10.py --model resnet_nobatchnorm_101
python test_cifar.py --model resnet_nobatchnorm_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_nobatchnorm_101.txt
python cifar10.py --model resnet_nobatchnorm_152
python test_cifar.py --model resnet_nobatchnorm_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_nobatchnorm_152.txt


