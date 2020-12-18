python mnist.py --model FC --n_hidden 500
python test_mnist.py --model FC --n_hidden 500 > /vulcanscratch/songweig/logs/adv_pool/FC_500.txt
python mnist.py --model FC --n_hidden 1000
python test_mnist.py --model FC --n_hidden 1000 > /vulcanscratch/songweig/logs/adv_pool/FC_1000.txt
python mnist.py --model FC --n_hidden 5000
python test_mnist.py --model FC --n_hidden 5000 > /vulcanscratch/songweig/logs/adv_pool/FC_5000.txt
python mnist.py --model FC --n_hidden 10000
python test_mnist.py --model FC --n_hidden 10000 > /vulcanscratch/songweig/logs/adv_pool/FC_10000.txt