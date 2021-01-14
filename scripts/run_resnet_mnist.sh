# python mnist.py --model resnet_18
# python test_mnist.py --model resnet_18 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_18.txt
# python mnist.py --model resnet_max_pooling_18
# python test_mnist.py --model resnet_max_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_max_pooling_18.txt
# python mnist.py --model resnet_34
# python test_mnist.py --model resnet_34 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_34.txt
# python mnist.py --model resnet_max_pooling_34
# python test_mnist.py --model resnet_max_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_max_pooling_34.txt
python mnist.py --model resnet_fc_pooling_18
python test_mnist.py --model resnet_fc_pooling_18 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_fc_pooling_18.txt
python mnist.py --model resnet_fc_pooling_34
python test_mnist.py --model resnet_fc_pooling_34 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_fc_pooling_34.txt
python mnist.py --model resnet_fc_pooling_50
python test_mnist.py --model resnet_fc_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_fc_pooling_50.txt
python mnist.py --model resnet_50
python test_mnist.py --model resnet_50 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_50.txt
python mnist.py --model resnet_max_pooling_50
python test_mnist.py --model resnet_max_pooling_50 > /vulcanscratch/songweig/logs/adv_pool/mnist/resnet_max_pooling_50.txt