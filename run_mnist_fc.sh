python mnist.py --model FC --n_hidden 5
python test_mnist.py --model FC --n_hidden 5 > /vulcanscratch/songweig/logs/adv_pool/FC_5.txt
python mnist.py --model FC --n_hidden 10
python test_mnist.py --model FC --n_hidden 10 > /vulcanscratch/songweig/logs/adv_pool/FC_10.txt
python mnist.py --model FC --n_hidden 50
python test_mnist.py --model FC --n_hidden 50 > /vulcanscratch/songweig/logs/adv_pool/FC_50.txt
python mnist.py --model FC --n_hidden 100
python test_mnist.py --model FC --n_hidden 100 > /vulcanscratch/songweig/logs/adv_pool/FC_100.txt