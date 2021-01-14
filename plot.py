import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt

log_dir_imagenet = '/vulcanscratch/songweig/logs/adv_pool/imagenet_unnorm'
log_dir_antialiased_imagenet = '/vulcanscratch/songweig/logs/adv_pool/antialiased_imagenet_unnorm'

# L2_acc = {'alexnet':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 'resnet18':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 
# 		'resnet34':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 'resnet50':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 
# 		'resnet101':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 'resnet152':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}}
# Linf_acc = {'alexnet':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 'resnet18':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 
# 		'resnet34':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 'resnet50':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 
# 		'resnet101':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 'resnet152':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}}

L2_acc = {'vgg11':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}, 'vgg13':{'0.55':0., '1.09':0., '2.18':0., '4.37':0.}}
Linf_acc = {'vgg11':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}, 'vgg13':{'0.01':0., '0.02':0., '0.03':0., '0.07':0.}}
clean_acc = {'vgg11':0., 'vgg13':0.}

strength_norm = {'L_2': {'0.55': '0.125', '1.09': '0.25', '2.18': '0.5', '4.37': '1'}, 'L_inf': {'0.01': '0.5', '0.02': '1', '0.03': '2', '0.07': '4'}}

base_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}
antialiased_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}

for model_name in L2_acc:
	with open(os.path.join(log_dir_imagenet, model_name+'.txt')) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in base_cnns[attack][model_name]:
				continue
			base_cnns[attack][model_name][strength] = float(acc)
	with open(os.path.join(log_dir_imagenet, model_name+'_bn.txt')) as f:
	# with open(os.path.join(log_dir_antialiased_imagenet, model_name+'.txt')) as f:
		final_run = False
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in base_cnns[attack][model_name]:
				continue
			antialiased_cnns[attack][model_name][strength] = float(acc)


########################################################################################################################################################
### draw vanilla CNNs v.s. antialiased CNNs
########################################################################################################################################################


n_models = len(L2_acc)
for attack in base_cnns:
	for strength in base_cnns[attack]['alexnet']:
		plt.clf()
		plt.scatter(np.arange(n_models), [base_cnns[attack][model_name][strength] for model_name in L2_acc], label='base CNNs', color='orange')
		plt.scatter(np.arange(n_models), [antialiased_cnns[attack][model_name][strength] for model_name in L2_acc], label='antialiased CNNs', color='turquoise')
		plt.xticks(np.arange(n_models), L2_acc.keys())
		plt.legend(loc='upper right')
		plt.title('test accuracy under %s adversarial attack with radius %s'%(attack, strength_norm[attack][strength]))
		plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/%s_%s.png'%(attack, strength_norm[attack][strength]))



########################################################################################################################################################
### draw vanilla VGG v.s. BN VGG
########################################################################################################################################################

for attack in list(gap_cnns.keys())[:-1]:
	epss = gap_cnns[attack]['vgg11']
	n_eps = len(epss)
	plt.clf()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, L2_acc.keys()):
		plt.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[gap_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name, color=color2)
		plt.plot(np.arange(n_eps+1), [clean_accs[model_name+'_bn'][0]]+[np_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_bn', color=color1)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/vgg_%s_%s.png'%(attack, strength_norm[attack][strength]))


########################################################################################################################################################
### draw clean accuracy
########################################################################################################################################################

clean_accs = {'alexnet': [56.55, 56.94], 'vgg11': [69.02, 70.51], 'vgg13': [69.93, 71.52], 'vgg16': [71.59, 72.96], 'vgg19': [72.38, 73.54], 
			'vgg11_bn': [70.38, 72.63], 'vgg13_bn': [71.55, 73.61], 'vgg16_bn': [73.36, 75.13], 'vgg19_bn': [74.24, 75.68], 'resnet18': [69.74, 71.67], 
			'resnet34': [73.30, 74.60], 'resnet50': [76.16, 77.41], 'resnet101': [77.37, 78.38], 'resnet152': [78.31, 79.07], 'resnext50_32x4d': [77.62, 77.93], 
			'resnext101_32x8d': [79.31, 79.33], 'wide_resnet50_2': [78.47, 78.70], 'wide_resnet101_2': [78.85, 78.99], 'densenet121': [74.43, 75.79], 
			'densenet169': [75.60, 76.73], 'densenet201': [76.90, 77.31], 'densenet161': [77.14, 77.88], 'mobilenet_v2': [71.88, 72.72]}



plt.clf()
plt.scatter(np.arange(n_models), [clean_accs[model_name][0] for model_name in L2_acc], label='base CNNs', color='orange')
plt.scatter(np.arange(n_models), [clean_accs[model_name][1] for model_name in L2_acc], label='antialiased CNNs', color='turquoise')
plt.xticks(np.arange(n_models), L2_acc.keys())
plt.legend(loc='upper right')
plt.title('clean test accuracy')
plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/clean.png')



########################################################################################################################################################
### draw clean accuracy
########################################################################################################################################################


strength_norm = {'L_2': {'0.55': '0.125', '1.09': '0.25', '2.18': '0.5', '4.37': '1'}, 'L_inf': {'0.01': '0.5/255', '0.02': '1/255', '0.03': '2/255', '0.07': '4/255'}}
colors = ['#003f5c', '#ffa600', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', 'green']
for attack in base_cnns:
	epss = base_cnns[attack]['alexnet']
	n_eps = len(epss)
	plt.clf()
	for color, model_name in zip(colors[::-1], L2_acc.keys()):
		# plt.scatter(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], label=model_name, color=color)
		plt.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name, color=color)
	plt.xticks(np.arange(n_eps+1), [0]+[strength_norm[attack][eps] for eps in list(epss.keys())])
	plt.legend(loc='upper right', ncol=3)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/%s_models.png'%(attack))


########################################################################################################################################################
### draw VGG on cifar w/ and w/o global pooling
########################################################################################################################################################

CAND_COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]


log_dir_cifar = '/vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug'

L2_acc = {'vgg11':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'vgg13':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 
		'vgg16':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'vgg19':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}}
Linf_acc = {'vgg11':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'vgg13':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 
		'vgg16':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'vgg19':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}}
clean_acc = {'vgg11':0., 'vgg13':0., 'vgg16':0., 'vgg19':0.}

gap_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}
np_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}

for model_name in L2_acc:
	with open(os.path.join(log_dir_cifar, model_name+'.txt')) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in gap_cnns[attack][model_name]:
				continue
			gap_cnns[attack][model_name][strength] = float(acc)
	with open(os.path.join(log_dir_cifar, model_name+'_nopooling.txt')) as f:
		final_run = False
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in np_cnns[attack][model_name]:
				continue
			np_cnns[attack][model_name][strength] = float(acc)


for attack in list(gap_cnns.keys())[:-1]:
	epss = gap_cnns[attack]['vgg11']
	n_eps = len(epss)
	plt.clf()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, L2_acc.keys()):
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[gap_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_gap', color=color2)
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[np_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_nop', color=color1)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_VGGs.png'%(attack))





########################################################################################################################################################
### draw AlexNet on cifar w/ and w/o global pooling
########################################################################################################################################################

CAND_COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]


log_dir_cifar = '/vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug'

L2_acc = {'alexnet':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}}
Linf_acc = {'alexnet':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}}
clean_acc = {'alexnet':0., 'vgg13':0., 'vgg16':0., 'vgg19':0.}

gap_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}
np_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}

for model_name in L2_acc:
	with open(os.path.join(log_dir_cifar, model_name+'.txt')) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in gap_cnns[attack][model_name]:
				continue
			gap_cnns[attack][model_name][strength] = float(acc)
	with open(os.path.join(log_dir_cifar, model_name+'_nopooling.txt')) as f:
		final_run = False
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in np_cnns[attack][model_name]:
				continue
			np_cnns[attack][model_name][strength] = float(acc)


for attack in list(gap_cnns.keys())[:-1]:
	epss = gap_cnns[attack]['alexnet']
	n_eps = len(epss)
	plt.clf()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, L2_acc.keys()):
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[gap_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_gap', color=color2)
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[np_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_nop', color=color1)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_alexnet.png'%(attack))



########################################################################################################################################################
### draw ResNet on cifar w/ and w/o global pooling
########################################################################################################################################################

CAND_COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]


log_dir_cifar = '/vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug'

L2_acc = {'resnet_18':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'resnet_34':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 
		'resnet_50':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'resnet_101':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}}
Linf_acc = {'resnet_18':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_34':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 
		'resnet_50':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}}
clean_acc = {'resnet_18':0., 'resnet_34':0., 'resnet_50':0., 'resnet_101':0.}

gap_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}
np_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}

for model_name in L2_acc:
	with open(os.path.join(log_dir_cifar, 'resnet_%s.txt'%(model_name.split('_')[1]))) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in gap_cnns[attack][model_name]:
				continue
			gap_cnns[attack][model_name][strength] = float(acc)

			
for model_name in L2_acc:
	with open(os.path.join(log_dir_cifar, 'resnet_no_pooling_%s.txt'%(model_name.split('_')[1]))) as f:
		final_run = False
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				gap_cnns['clean'][model_name] = float(acc)
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in np_cnns[attack][model_name]:
				continue
			np_cnns[attack][model_name][strength] = float(acc)


for attack in list(gap_cnns.keys())[:-1]:
	epss = gap_cnns[attack]['resnet_18']
	n_eps = len(epss)
	plt.clf()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, gap_cnns[attack].keys()):
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[gap_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_gap', color=color2)
		plt.plot(np.arange(n_eps+1), [gap_cnns['clean'][model_name]]+[np_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_nop', color=color1)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_ResNets.png'%(attack))

