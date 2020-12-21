python mnist.py --model Conv_linear_max --n_hidden 5
python test_mnist.py --model Conv_linear_max --n_hidden 5 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_5.txt
python mnist.py --model Conv_linear_max --n_hidden 10
python test_mnist.py --model Conv_linear_max --n_hidden 10 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_10.txt
python mnist.py --model Conv_linear_max --n_hidden 50
python test_mnist.py --model Conv_linear_max --n_hidden 50 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_50.txt
python mnist.py --model Conv_linear_max --n_hidden 100
python test_mnist.py --model Conv_linear_max --n_hidden 100 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_100.txt
python mnist.py --model Conv_linear_max --n_hidden 500
python test_mnist.py --model Conv_linear_max --n_hidden 500 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_500.txt
python mnist.py --model Conv_linear_max --n_hidden 1000
python test_mnist.py --model Conv_linear_max --n_hidden 1000 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_1000.txt
python mnist.py --model Conv_linear_max --n_hidden 5000
python test_mnist.py --model Conv_linear_max --n_hidden 5000 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_5000.txt
python mnist.py --model Conv_linear_max --n_hidden 10000
python test_mnist.py --model Conv_linear_max --n_hidden 10000 > /vulcanscratch/songweig/logs/adv_pool/Conv_linear_max_10000.txt