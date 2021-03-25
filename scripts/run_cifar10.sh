
python test_cifar.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_18.txt
python test_cifar.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_34.txt
python test_cifar.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_50.txt
python test_cifar.py --model resnet_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_101.txt
python test_cifar.py --model resnet_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_152.txt


python test_cifar.py --model resnet_max_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_max_pooling_PGD_18.txt
python test_cifar.py --model resnet_max_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_max_pooling_PGD_34.txt
python test_cifar.py --model resnet_max_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_max_pooling_PGD_50.txt
python test_cifar.py --model resnet_max_pooling_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_max_pooling_PGD_101.txt
python test_cifar.py --model resnet_max_pooling_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_max_pooling_PGD_152.txt



python cifar10.py --model alexnet
python test_cifar.py --model alexnet > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet.txt
python cifar10.py --model alexnet_nopooling
python test_cifar.py --model alexnet_nopooling > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling.txt

python cifar10.py --model VGG11
python test_cifar.py --model VGG11 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg11.txt
python cifar10.py --model VGG13
python test_cifar.py --model VGG13 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg13.txt
python cifar10.py --model VGG16
python test_cifar.py --model VGG16 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg16.txt
python cifar10.py --model VGG19
python test_cifar.py --model VGG19 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg19.txt


python cifar10.py --model VGG11_NoPooling
python test_cifar.py --model VGG11_NoPooling > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg11_nopooling.txt
python cifar10.py --model VGG13_NoPooling
python test_cifar.py --model VGG13_NoPooling > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg13_nopooling.txt
python cifar10.py --model VGG16_NoPooling
python test_cifar.py --model VGG16_NoPooling > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg16_nopooling.txt
python cifar10.py --model VGG19_NoPooling
python test_cifar.py --model VGG19_NoPooling > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/vgg19_nopooling.txt


python cifar10.py --model resnet_nopooling_34
python test_cifar.py --model resnet_nopooling_34 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling_34.txt
python cifar10.py --model resnet_nopooling_152
python test_cifar.py --model resnet_nopooling_152 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling_152.txt

python cifar10.py --model resnet_nopooling_18
python test_cifar.py --model resnet_nopooling_18 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling_18.txt
python cifar10.py --model resnet_nopooling_50
python test_cifar.py --model resnet_nopooling_50 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling_50.txt
python cifar10.py --model resnet_nopooling_101
python test_cifar.py --model resnet_nopooling_101 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/resnet_no_pooling_1