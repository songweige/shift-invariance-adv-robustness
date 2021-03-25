python cifar10.py --model resnet_nopooling2_34
python test_cifar.py --model resnet_nopooling2_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling2_34.txt
python cifar10.py --model resnet_nopooling2_18
python test_cifar.py --model resnet_nopooling2_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling2_18.txt
python cifar10.py --model resnet_nopooling2_50
python test_cifar.py --model resnet_nopooling2_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling2_50.txt
python cifar10.py --model resnet_nopooling2_101
python test_cifar.py --model resnet_nopooling2_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling2_101.txt
python cifar10.py --model resnet_nopooling2_152
python test_cifar.py --model resnet_nopooling2_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling2_152.txt
