n_hiddens = [5, 10, 50, 100, 500, 1000, 5000, 10000]
for n_hidden in n_hiddens[::-1]:
	# with open('FC_linear_%d.txt'%n_hidden) as f:
	# 	lines = f.readlines()
	# 	print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))
	# with open('Conv_linear_%d.txt'%n_hidden) as f:
	# 	lines = f.readlines()
	# 	print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))
	print(0)
	with open('Conv_ks3_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))
	with open('Conv_max_ks3_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))



for n_hidden in n_hiddens[::-1]:
	with open('Conv_max_ks3_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(lines[1].rstrip('%\n').split()[-1])
		print('\n')



n_hiddens = [5, 10, 50, 100, 500, 1000, 5000]
n_hiddens = [10000]
for n_hidden in n_hiddens[::-1]:
	with open('FC_linear_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))
	with open('Conv_linear_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))
	with open('Conv_linear_max_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		print(' '.join([line.rstrip('%\n').split()[-1] for line in lines[2:]]))



############################################################################################
### Nice print results of ResNet on Cifar 10
############################################################################################
n_hiddens = [18, 34, 50]
n_hiddens = [18, 34, 50, 101, 152]
for n_hidden in n_hiddens[::-1]:
	with open('resnet_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[4:5]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:10]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[10:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))
	with open('resnet_max_pooling_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[4:5]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:10]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[10:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))


for n_hidden in n_hiddens[::-1]:
	with open('mnist/resnet_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[4:5]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:10]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[10:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))
	with open('mnist/resnet_max_pooling_%d.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[4:5]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:10]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[10:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))
		


n_hiddens = [28]
for n_hidden in n_hiddens[::-1]:
	with open('resnet_zero_padding/wide_resnet_%d_10.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:6]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[6:11]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[11:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))
	with open('resnet_zero_padding/wide_resnet_max_pooling_%d_10.txt'%n_hidden) as f:
		lines = f.readlines()
		clean_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[5:6]])
		l2_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[6:11]])
		linf_acc = ' '.join([line.rstrip('%\n').split()[-1] for line in lines[11:]])
		print('  '.join([clean_acc, l2_acc, linf_acc]))



for n_hidden in n_hiddens[::-1]:
	with open('resnet_zero_padding/wide_resnet_%d_10.txt'%n_hidden) as f:
		lines = f.readlines()
		print(lines[2].rstrip('%\n').split()[-1])
		print('\n')