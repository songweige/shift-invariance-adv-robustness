python mnist.py --model Conv --n_hidden 500
python test_mnist.py --model Conv --n_hidden 500 > /vulcanscratch/songweig/logs/adv_pool/Conv_500.txt
python mnist.py --model Conv --n_hidden 1000
python test_mnist.py --model Conv --n_hidden 1000 > /vulcanscratch/songweig/logs/adv_pool/Conv_1000.txt
python mnist.py --model Conv --n_hidden 5000
python test_mnist.py --model Conv --n_hidden 5000 > /vulcanscratch/songweig/logs/adv_pool/Conv_5000.txt
python mnist.py --model Conv --n_hidden 10000
python test_mnist.py --model Conv --n_hidden 10000 > /vulcanscratch/songweig/logs/adv_pool/Conv_10000.txt