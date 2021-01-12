import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt

log_dir_imagenet = '/vulcanscratch/songweig/logs/adv_pool/imagenet'
log_dir_antialiased_imagenet = '/vulcanscratch/songweig/logs/adv_pool/antialiased_imagenet'

L2_acc = {'alexnet':{'0.50':0., '1.00':0., '2.00':0.}, 'resnet18':{'0.50':0., '1.00':0., '2.00':0.}, 
		'resnet34':{'0.50':0., '1.00':0., '2.00':0.}, 'resnet50':{'0.50':0., '1.00':0., '2.00':0.}, 
		'resnet101':{'0.50':0., '1.00':0., '2.00':0.}, 'resnet152':{'0.50':0., '1.00':0., '2.00':0.}}
Linf_acc = {'alexnet':{'0.02':0., '0.03':0.}, 'resnet18':{'0.02':0., '0.03':0.}, 
		'resnet34':{'0.02':0., '0.03':0.}, 'resnet50':{'0.02':0., '0.03':0.}, 
		'resnet101':{'0.02':0., '0.03':0.}, 'resnet152':{'0.02':0., '0.03':0.}}

base_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}
antialiased_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}

for model_name in L2_acc:
	with open(os.path.join(log_dir_imagenet, model_name+'_train_10000.txt')) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			attack = re.search(r'\((.*?), eps', line).group(1)
			strength = re.search(r'eps=(.*?)\)', line).group(1)
			acc = re.search(r': (.*?)\%', line).group(1)
			if strength not in base_cnns[attack][model_name]:
				continue
			base_cnns[attack][model_name][strength] = float(acc)
	with open(os.path.join(log_dir_antialiased_imagenet, model_name+'_train_10000.txt')) as f:
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


n_models = len(L2_acc)
for attack in base_cnns:
	for strength in base_cnns[attack]['alexnet']:
		plt.clf()
		plt.scatter(np.arange(n_models), [base_cnns[attack][model_name][strength] for model_name in L2_acc], label='base CNNs', color='orange')
		plt.scatter(np.arange(n_models), [antialiased_cnns[attack][model_name][strength] for model_name in L2_acc], label='antialiased CNNs', color='turquoise')
		plt.xticks(np.arange(n_models), L2_acc.keys())
		plt.legend(loc='upper right')
		plt.title('test accuracy under %s adversarial attack with radius %s'%(attack, strength))
		plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/%s_%s_train_10000.png'%(attack, strength))





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




colors = ['#003f5c', '#ffa600', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', 'green']
for attack in base_cnns:
	epss = base_cnns[attack]['alexnet']
	n_eps = len(epss)
	plt.clf()
	for color, model_name in zip(colors[::-1], L2_acc.keys()):
		plt.scatter(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], label=model_name, color=color)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	plt.legend(loc='upper right', ncol=3)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/%s_models.png'%(attack))
