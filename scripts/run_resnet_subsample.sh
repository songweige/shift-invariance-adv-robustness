python cifar10.py --model resnet_18 --n_data 1000
python test_cifar.py --model resnet_18 --n_data 1000 > /vulcanscratch/songweig/logs/adv_pool/resnet_18_1000.txt
python cifar10.py --model resnet_34 --n_data 1000
python test_cifar.py --model resnet_34 --n_data 1000 > /vulcanscratch/songweig/logs/adv_pool/resnet_34_1000.txt
python cifar10.py --model resnet_50 --n_data 1000
python test_cifar.py --model resnet_50 --n_data 1000 > /vulcanscratch/songweig/logs/adv_pool/resnet_50_1000.txt
python cifar10.py --model resnet_18 --n_data 5000
python test_cifar.py --model resnet_18 --n_data 5000 > /vulcanscratch/songweig/logs/adv_pool/resnet_18_5000.txt
python cifar10.py --model resnet_34 --n_data 5000
python test_cifar.py --model resnet_34 --n_data 5000 > /vulcanscratch/songweig/logs/adv_pool/resnet_34_5000.txt
python cifar10.py --model resnet_50 --n_data 5000
python test_cifar.py --model resnet_50 --n_data 5000 > /vulcanscratch/songweig/logs/adv_pool/resnet_50_5000.txt
python cifar10.py --model resnet_18 --n_data 10000
python test_cifar.py --model resnet_18 --n_data 10000 > /vulcanscratch/songweig/logs/adv_pool/resnet_18_10000.txt
python cifar10.py --model resnet_34 --n_data 10000
python test_cifar.py --model resnet_34 --n_data 10000 > /vulcanscratch/songweig/logs/adv_pool/resnet_34_10000.txt
python cifar10.py --model resnet_50 --n_data 10000
python test_cifar.py --model resnet_50 --n_data 10000 > /vulcanscratch/songweig/logs/adv_pool/resnet_50_10000.txt