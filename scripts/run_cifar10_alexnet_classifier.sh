python cifar10.py --model alexnet_linear
python test_cifar.py --model alexnet_linear > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_linear_gap.txt
python cifar10.py --model alexnet_MLP2_8192
python test_cifar.py --model alexnet_MLP2_8192 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP2_8192.txt
python cifar10.py --model alexnet_MLP2_1024
python test_cifar.py --model alexnet_MLP2_1024 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP2_1024.txt
python cifar10.py --model alexnet_MLP2_512
python test_cifar.py --model alexnet_MLP2_512 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP2_512.txt


python cifar10.py --model alexnet_MLP_8192
python test_cifar.py --model alexnet_MLP_8192 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP_8192.txt
python cifar10.py --model alexnet_MLP_4096
python test_cifar.py --model alexnet_MLP_4096 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP_4096.txt
python cifar10.py --model alexnet_MLP_1024
python test_cifar.py --model alexnet_MLP_1024 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP_1024.txt
python cifar10.py --model alexnet_MLP_512
python test_cifar.py --model alexnet_MLP_512 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug_classifier/alexnet_MLP_512.txt