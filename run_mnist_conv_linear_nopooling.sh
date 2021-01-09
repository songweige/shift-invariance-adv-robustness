python mnist.py --model Conv_linear_nopooling --n_hidden 4
python test_mnist.py --model Conv_linear_nopooling --n_hidden 4 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_4.txt
python mnist.py --model Conv_linear_nopooling --n_hidden 8
python test_mnist.py --model Conv_linear_nopooling --n_hidden 8 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_8.txt
python mnist.py --model Conv_linear_nopooling --n_hidden 42
python test_mnist.py --model Conv_linear_nopooling --n_hidden 42 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_42.txt
python mnist.py --model Conv_linear_nopooling --n_hidden 85
python test_mnist.py --model Conv_linear_nopooling --n_hidden 85 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_85.txt
python mnist.py --model Conv_linear_nopooling --n_hidden 426
python test_mnist.py --model Conv_linear_nopooling --n_hidden 426 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_426.txt
python mnist.py --model Conv_linear_nopooling --n_hidden 853
python test_mnist.py --model Conv_linear_nopooling --n_hidden 853 > /vulcanscratch/songweig/logs/adv_pool/mnist/simple_ks28_padding14/Conv_linear_nopooling_853.txt