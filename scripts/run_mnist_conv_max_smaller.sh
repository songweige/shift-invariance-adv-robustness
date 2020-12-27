python mnist.py --model Conv_max --n_hidden 5 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 5 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_5.txt
python mnist.py --model Conv_max --n_hidden 10 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 10 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_10.txt
python mnist.py --model Conv_max --n_hidden 50 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 50 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_50.txt
python mnist.py --model Conv_max --n_hidden 100 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 100 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_100.txt
python mnist.py --model Conv_max --n_hidden 500 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 500 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_500.txt
python mnist.py --model Conv_max --n_hidden 1000 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 1000 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_1000.txt
python mnist.py --model Conv_max --n_hidden 5000 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 5000 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_5000.txt
python mnist.py --model Conv_max --n_hidden 10000 --kernel_size 3
python test_mnist.py --model Conv_max --n_hidden 10000 --kernel_size 3 > /vulcanscratch/songweig/logs/adv_pool/Conv_max_ks3_10000.txt