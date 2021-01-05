python cifar10.py --model resnet_18
python test_cifar.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_18.txt
python cifar10.py --model resnet_max_pooling_18
python test_cifar.py --model resnet_max_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_max_pooling_18.txt
python cifar10.py --model resnet_34
python test_cifar.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_34.txt
python cifar10.py --model resnet_max_pooling_34
python test_cifar.py --model resnet_max_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_max_pooling_34.txt
python cifar10.py --model resnet_50
python test_cifar.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_50.txt
python cifar10.py --model resnet_max_pooling_50
python test_cifar.py --model resnet_max_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_max_pooling_50.txt