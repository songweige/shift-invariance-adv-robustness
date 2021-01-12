# python cifar10.py --model resnet_18
# python test_cifar.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/resnet_18.txt
python test_cifar.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/resnet_PGD_18.txt
# python cifar10.py --model resnet_34
# python test_cifar.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/resnet_34.txt
python test_cifar.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/resnet_PGD_34.txt
# python cifar10.py --model resnet_50
# python test_cifar.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/resnet_50.txt
python test_cifar.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/resnet_PGD_50.txt
# python cifar10.py --model resnet_101
# python test_cifar.py --model resnet_101 > /vulcanscratch/songweig/logs/adv_pool/resnet_101.txt
python test_cifar.py --model resnet_101 > /vulcanscratch/songweig/logs/adv_pool/resnet_PGD_101.txt
# python cifar10.py --model resnet_152
# python test_cifar.py --model resnet_152 > /vulcanscratch/songweig/logs/adv_pool/resnet_152.txt
python test_cifar.py --model resnet_152 > /vulcanscratch/songweig/logs/adv_pool/resnet_PGD_152.txt



python cifar10.py --model alexnet
python test_cifar.py --model alexnet > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet.txt

python test_cifar.py --model VGG13 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg13.txt
python cifar10.py --model VGG11
python test_cifar.py --model VGG11 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg11.txt
python cifar10.py --model VGG16
python test_cifar.py --model VGG16 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg16.txt
