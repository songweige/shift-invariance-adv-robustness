# python cifar10.py --model wide_resnet_28_10
# python test_cifar.py --model wide_resnet_28_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_28_10.txt
python test_cifar.py --model wide_resnet_28_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_PGD_28_10.txt
# python cifar10.py --model wide_resnet_40_10
# python test_cifar.py --model wide_resnet_40_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_40_10.txt
python test_cifar.py --model wide_resnet_40_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_PGD_40_10.txt