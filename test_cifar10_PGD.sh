python cifar10.py --model resnet_max_pooling_18
python test_cifar.py --model resnet_max_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_max_pooling_PGD_18.txt
python test_cifar.py --model resnet_max_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_max_pooling_PGD_34.txt
python test_cifar.py --model resnet_max_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_max_pooling_PGD_50.txt
python test_cifar.py --model resnet_max_pooling_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_max_pooling_PGD_101.txt
python test_cifar.py --model resnet_max_pooling_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_max_pooling_PGD_152.txt


python test_cifar.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_PGD_18.txt
python test_cifar.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_PGD_34.txt
python test_cifar.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_PGD_50.txt
python test_cifar.py --model resnet_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_PGD_101.txt
python test_cifar.py --model resnet_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10_PGD/resnet_PGD_152.txt