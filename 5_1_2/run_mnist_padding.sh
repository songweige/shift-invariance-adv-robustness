python mnist.py --model Conv --n_hidden 1000 --padding_size 0
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 0 > ./logs/adv_pool/Conv_1000_padding0.txt
python mnist.py --model Conv --n_hidden 1000 --padding_size 2
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 2 > ./logs/adv_pool/Conv_1000_padding2.txt

python mnist.py --model Conv --n_hidden 1000 --padding_size 4
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 4 > ./logs/adv_pool/Conv_1000_padding4.txt
python mnist.py --model Conv --n_hidden 1000 --padding_size 6
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 6 > ./logs/adv_pool/Conv_1000_padding6.txt

python mnist.py --model Conv --n_hidden 1000 --padding_size 8
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 8 > ./logs/adv_pool/Conv_1000_padding8.txt
python mnist.py --model Conv --n_hidden 1000 --padding_size 10
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 10 > ./logs/adv_pool/Conv_1000_padding10.txt

python mnist.py --model Conv --n_hidden 1000 --padding_size 12
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 12 > ./logs/adv_pool/Conv_1000_padding12.txt
python mnist.py --model Conv --n_hidden 1000 --padding_size 14
python test_mnist.py --model Conv --n_hidden 1000 --padding_size 14 > ./logs/adv_pool/Conv_1000_padding14.txt
