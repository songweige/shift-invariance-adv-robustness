import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

########################################################################################################################################################
### draw MNIST for double descent
########################################################################################################################################################

CAND_COLORS = ['#fb9a99','#e31a1c','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]


log_dir_mnist = '/vulcanscratch/songweig/logs/double_descent'
log_dir_mnist = '/vulcanscratch/songweig/logs/double_descent_4k'
model_names = sorted([int(fn.split('_')[1].split('.')[0]) for fn in os.listdir(log_dir_mnist)])[:-3]

# attack_params = {'L_2': [32/256., 64./256., 128./256, 256/256, 1.5], 'L_inf': [1/256., 2/256., 4/256., 8/256., 16/256.]}
attack_params = {'L_2': ['16/255', '32/255', '64/255', '128/255', '192/255'], 'L_inf': ['0.5/255', '1/255', '2/255', '4/255', '8/255']}
attack_params = {'L_2': ['0.25', '0.5', '0.75', '0.1'], 'L_inf': ['1/255', '2/255', '4/255', '8/255']}
L2_acc = {model_name:{'0.50':0., '1.00':0., '1.50':0., '2.00':0., '2.50':0.} for model_name in model_names}
Linf_acc = {model_name:{'0.05':0., '0.10':0., '0.15':0., '0.20':0., '0.25':0.} for model_name in model_names}
train_acc = {model_name:0. for model_name in model_names}
test_acc = {model_name:0. for model_name in model_names}
cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}

for model_name in model_names:
	with open(os.path.join(log_dir_mnist, 'FC_%d.txt'%(model_name))) as f:
		for line in f:
			if line.startswith('Training'):
				acc = re.search(r'Acc: (.*?)\%', line).group(1)
				train_acc[model_name] = float(acc)
			if line.startswith('Test'):
				acc = re.search(r'Acc: (.*?)\%', line).group(1)
				test_acc[model_name] = float(acc)
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in cnns[attack][model_name]:
				continue
			cnns[attack][model_name][strength] = float(acc)


font = {'family' : 'serif',
 		'serif':   'Times',
        'size'   : 15}

matplotlib.rc('font', **font)

for attack in list(cnns.keys()):
	epss = cnns[attack][model_names[0]]
	n_eps = len(epss)
	n_model = len(model_names)
	plt.clf()
	fig = plt.figure()
	ax = plt.subplot(111)
	plt.plot(model_names, [train_acc[model_name] for model_name in model_names], marker='o', label='train', color=colors2[0])
	plt.plot(model_names, [test_acc[model_name] for model_name in model_names], marker='o', label='test', color=colors2[1])
	for color1, color2, strength in zip(colors1, colors2[2:], epss):
		plt.plot(model_names, [cnns[attack][model_name][strength] for model_name in model_names], marker='o', label=strength, color=color2)
	# plt.ylim(5, 92)
	# plt.xlabel('epsilon')
	# plt.ylabel('accuracy')
	box = ax.get_position()
	plt.tight_layout()
	ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=4)
	# plt.savefig('/vulcanscratch/songweig/plots/double_descent/%s_MNIST_full.png'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/double_descent/%s_MNIST_4k.png'%(attack))


########################################################################################################################################################
### better draw AlexNet v.s. ResNet on cifar
########################################################################################################################################################

log_dir_mnist = '/vulcanscratch/songweig/logs/double_descent'
log_dir_mnist = '/vulcanscratch/songweig/logs/double_descent_4k'
model_names = sorted([int(fn.split('_')[1].split('.')[0]) for fn in os.listdir(log_dir_mnist)])[:-3]

# attack_params = {'L_2': [32/256., 64./256., 128./256, 256/256, 1.5], 'L_inf': [1/256., 2/256., 4/256., 8/256., 16/256.]}
attack_params = {'L_2': ['16/255', '32/255', '64/255', '128/255', '192/255'], 'L_inf': ['0.5/255', '1/255', '2/255', '4/255', '8/255']}
attack_params = {'L_2': ['0.25', '0.5', '0.75', '0.1'], 'L_inf': ['1/255', '2/255', '4/255', '8/255']}
L2_acc = {model_name:{'0.50':0., '1.00':0., '1.50':0., '2.00':0., '2.50':0.} for model_name in model_names}
Linf_acc = {model_name:{'0.05':0., '0.10':0., '0.15':0., '0.20':0., '0.25':0.} for model_name in model_names}
train_acc = {model_name:0. for model_name in model_names}
test_acc = {model_name:0. for model_name in model_names}
cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}

for model_name in model_names:
	with open(os.path.join(log_dir_mnist, 'FC_%d.txt'%(model_name))) as f:
		for line in f:
			if line.startswith('Training'):
				acc = re.search(r'Loss: (.*?) \|', line).group(1)
				train_acc[model_name] = float(acc)
			if line.startswith('Test'):
				acc = re.search(r'Loss: (.*?) \|', line).group(1)
				test_acc[model_name] = float(acc)
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r'Loss: (.*?)\n', line).group(1)
			if strength not in cnns[attack][model_name]:
				continue
			cnns[attack][model_name][strength] = float(acc)


font = {'family' : 'serif',
 		'serif':   'Times',
        'size'   : 15}

matplotlib.rc('font', **font)

for attack in list(cnns.keys()):
	epss = cnns[attack][model_names[0]]
	n_eps = len(epss)
	n_model = len(model_names)
	plt.clf()
	fig = plt.figure()
	ax = plt.subplot(111)
	plt.plot(model_names, [train_acc[model_name] for model_name in model_names], marker='o', label='train', color=colors2[0])
	plt.plot(model_names, [test_acc[model_name] for model_name in model_names], marker='o', label='test', color=colors2[1])
	for color1, color2, strength in zip(colors1, colors2[2:], epss):
		plt.plot(model_names, [cnns[attack][model_name][strength] for model_name in model_names], marker='o', label=strength, color=color2)
	# plt.ylim(5, 92)
	# plt.xlabel('epsilon')
	# plt.ylabel('accuracy')
	box = ax.get_position()
	plt.tight_layout()
	ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=4)
	# plt.savefig('/vulcanscratch/songweig/plots/double_descent/%s_MNIST_full_loss.png'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/double_descent/%s_MNIST_4k_loss.png'%(attack))

