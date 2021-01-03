# python cifar10.py --model resnet_max_pooling_18
# python test_cifar.py --model resnet_max_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/resnet_max_pooling_18.txt
# python cifar10.py --model resnet_max_pooling_34
# python test_cifar.py --model resnet_max_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/resnet_max_pooling_34.txt
# python cifar10.py --model resnet_max_pooling_50
# python test_cifar.py --model resnet_max_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/resnet_max_pooling_50.txt
python cifar10.py --model resnet_max_pooling_101
python test_cifar.py --model resnet_max_pooling_101 > /vulcanscratch/songweig/logs/adv_pool/resnet_max_pooling_101.txt
python cifar10.py --model resnet_max_pooling_152
python test_cifar.py --model resnet_max_pooling_152 > /vulcanscratch/songweig/logs/adv_pool/resnet_max_pooling_152.txt