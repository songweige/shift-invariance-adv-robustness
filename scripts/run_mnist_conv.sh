python mnist.py --model Conv --n_hidden 5
python test_mnist.py --model Conv --n_hidden 5 > /vulcanscratch/songweig/logs/adv_pool/Conv_5.txt
python mnist.py --model Conv --n_hidden 10
python test_mnist.py --model Conv --n_hidden 10 > /vulcanscratch/songweig/logs/adv_pool/Conv_10.txt
python mnist.py --model Conv --n_hidden 50
python test_mnist.py --model Conv --n_hidden 50 > /vulcanscratch/songweig/logs/adv_pool/Conv_50.txt
python mnist.py --model Conv --n_hidden 100
python test_mnist.py --model Conv --n_hidden 100 > /vulcanscratch/songweig/logs/adv_pool/Conv_100.txt
python mnist.py --model Conv --n_hidden 500
python test_mnist.py --model Conv --n_hidden 500 > /vulcanscratch/songweig/logs/adv_pool/Conv_500.txt
python mnist.py --model Conv --n_hidden 1000
python test_mnist.py --model Conv --n_hidden 1000 > /vulcanscratch/songweig/logs/adv_pool/Conv_1000.txt
python mnist.py --model Conv --n_hidden 5000
python test_mnist.py --model Conv --n_hidden 5000 > /vulcanscratch/songweig/logs/adv_pool/Conv_5000.txt
python mnist.py --model Conv --n_hidden 10000
python test_mnist.py --model Conv --n_hidden 10000 > /vulcanscratch/songweig/logs/adv_pool/Conv_10000.txt