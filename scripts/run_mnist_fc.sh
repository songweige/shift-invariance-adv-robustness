python mnist.py --model FC --n_hidden 5
python test_mnist.py --model FC --n_hidden 5 > /vulcanscratch/songweig/logs/adv_pool/FC_5.txt
python mnist.py --model FC --n_hidden 10
python test_mnist.py --model FC --n_hidden 10 > /vulcanscratch/songweig/logs/adv_pool/FC_10.txt
python mnist.py --model FC --n_hidden 50
python test_mnist.py --model FC --n_hidden 50 > /vulcanscratch/songweig/logs/adv_pool/FC_50.txt
python mnist.py --model FC --n_hidden 100
python test_mnist.py --model FC --n_hidden 100 > /vulcanscratch/songweig/logs/adv_pool/FC_100.txt
python mnist.py --model FC --n_hidden 500
python test_mnist.py --model FC --n_hidden 500 > /vulcanscratch/songweig/logs/adv_pool/FC_500.txt
python mnist.py --model FC --n_hidden 1000
python test_mnist.py --model FC --n_hidden 1000 > /vulcanscratch/songweig/logs/adv_pool/FC_1000.txt
python mnist.py --model FC --n_hidden 5000
python test_mnist.py --model FC --n_hidden 5000 > /vulcanscratch/songweig/logs/adv_pool/FC_5000.txt
python mnist.py --model FC --n_hidden 10000
python test_mnist.py --model FC --n_hidden 10000 > /vulcanscratch/songweig/logs/adv_pool/FC_10000.txt



python mnist.py --model FC --n_hidden 46
python test_mnist.py --model FC --n_hidden 46 > /vulcanscratch/songweig/logs/double_descent/FC_46.txt
python mnist.py --model FC --n_hidden 47
python test_mnist.py --model FC --n_hidden 47 > /vulcanscratch/songweig/logs/double_descent/FC_47.txt
python mnist.py --model FC --n_hidden 48
python test_mnist.py --model FC --n_hidden 48 > /vulcanscratch/songweig/logs/double_descent/FC_48.txt
python mnist.py --model FC --n_hidden 49
python test_mnist.py --model FC --n_hidden 49 > /vulcanscratch/songweig/logs/double_descent/FC_49.txt
python mnist.py --model FC --n_hidden 51
python test_mnist.py --model FC --n_hidden 51 > /vulcanscratch/songweig/logs/double_descent/FC_51.txt
python mnist.py --model FC --n_hidden 52
python test_mnist.py --model FC --n_hidden 52 > /vulcanscratch/songweig/logs/double_descent/FC_52.txt
python mnist.py --model FC --n_hidden 53
python test_mnist.py --model FC --n_hidden 53 > /vulcanscratch/songweig/logs/double_descent/FC_53.txt
python mnist.py --model FC --n_hidden 54
python test_mnist.py --model FC --n_hidden 54 > /vulcanscratch/songweig/logs/double_descent/FC_54.txt