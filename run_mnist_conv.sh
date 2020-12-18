python mnist.py --model Conv --n_hidden 5
python test_mnist.py --model Conv --n_hidden 5 > /vulcanscratch/songweig/logs/adv_pool/Conv_5.txt
python mnist.py --model Conv --n_hidden 10
python test_mnist.py --model Conv --n_hidden 10 > /vulcanscratch/songweig/logs/adv_pool/Conv_10.txt
python mnist.py --model Conv --n_hidden 50
python test_mnist.py --model Conv --n_hidden 50 > /vulcanscratch/songweig/logs/adv_pool/Conv_50.txt
python mnist.py --model Conv --n_hidden 100
python test_mnist.py --model Conv --n_hidden 100 > /vulcanscratch/songweig/logs/adv_pool/Conv_100.txt