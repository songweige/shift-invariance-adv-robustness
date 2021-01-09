# python cifar10.py --model wide_resnet_max_pooling_28_10
# python test_cifar.py --model wide_resnet_max_pooling_28_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_max_pooling_28_10.txt
python test_cifar.py --model wide_resnet_max_pooling_28_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_max_pooling_PGD_28_10.txt
# python cifar10.py --model wide_resnet_max_pooling_40_10
# python test_cifar.py --model wide_resnet_max_pooling_40_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_max_pooling_40_10.txt
python test_cifar.py --model wide_resnet_max_pooling_40_10 > /vulcanscratch/songweig/logs/adv_pool/wide_resnet_max_pooling_PGD_40_10.txt