python cifar10.py --model alexnet --n_data 100
python test_cifar.py --model alexnet --n_data 100 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_100.txt
python cifar10.py --model alexnet --n_data 500
python test_cifar.py --model alexnet --n_data 500 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_500.txt
python cifar10.py --model alexnet --n_data 1000
python test_cifar.py --model alexnet --n_data 1000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_1000.txt
python cifar10.py --model alexnet --n_data 5000
python test_cifar.py --model alexnet --n_data 5000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_5000.txt
python cifar10.py --model alexnet --n_data 10000
python test_cifar.py --model alexnet --n_data 10000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_10000.txt


python cifar10.py --model alexnet_nopooling --n_data 100
python test_cifar.py --model alexnet_nopooling --n_data 100 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling_100.txt
python cifar10.py --model alexnet_nopooling --n_data 500
python test_cifar.py --model alexnet_nopooling --n_data 500 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling_500.txt
python cifar10.py --model alexnet_nopooling --n_data 1000
python test_cifar.py --model alexnet_nopooling --n_data 1000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling_1000.txt
python cifar10.py --model alexnet_nopooling --n_data 5000
python test_cifar.py --model alexnet_nopooling --n_data 5000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling_5000.txt
python cifar10.py --model alexnet_nopooling --n_data 10000
python test_cifar.py --model alexnet_nopooling --n_data 10000 > /vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug/alexnet_nopooling_10000.txt
