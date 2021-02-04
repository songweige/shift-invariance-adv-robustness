import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

########################################################################################################################################################
### draw AlexNet v.s. ResNet on cifar
########################################################################################################################################################

CAND_COLORS = ['#fb9a99','#e31a1c','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]


log_dir_cifar = '/vulcanscratch/songweig/logs/adv_pool/cifar10/resnet_circular_padding_noaug'

# attack_params = {'L_2': [32/256., 64./256., 128./256, 256/256, 1.5], 'L_inf': [1/256., 2/256., 4/256., 8/256., 16/256.]}
attack_params = {'L_2': ['16/255', '32/255', '64/255', '128/255', '192/255'], 'L_inf': ['0.5/255', '1/255', '2/255', '4/255', '8/255']}
attack_params = {'L_2': ['0.25', '0.5', '0.75', '0.1'], 'L_inf': ['1/255', '2/255', '4/255', '8/255']}
# L2_acc = {'alexnet':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'resnet_18':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.},
# 		 'resnet_34':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'resnet_50':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.},
# 		 'resnet_101':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}, 'resnet_152':{'0.12':0., '0.25':0., '0.50':0., '1.00':0., '1.50':0.}}
# Linf_acc = {'alexnet':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_18':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 
# 		 'resnet_34':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_50':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.},
# 		 'resnet_101':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_152':{'0.00':0., '0.01':0., '0.02':0., '0.03':0., '0.06':0.}}
L2_acc = {'alexnet':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.}, 'resnet_18':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.},
		 'resnet_34':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.}, 'resnet_50':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.},
		 'resnet_101':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.}, 'resnet_152':{'0.50':0., '1.00':0., '1.50':0., '2.00':0.}}
Linf_acc = {'alexnet':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_18':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 
		 'resnet_34':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_50':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.},
		 'resnet_101':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.}, 'resnet_152':{'0.01':0., '0.02':0., '0.03':0., '0.06':0.}}
clean_acc = {'alexnet':0., 'resnet_18':0., 'resnet_34':0., 'resnet_50':0., 'resnet_101':0., 'resnet_152':0.}

cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc), 'clean': copy.deepcopy(clean_acc)}
model_names = {'alexnet': 'AlexNet', 'resnet_18': 'ResNet-18', 'resnet_34': 'ResNet-34', 'resnet_50': 'ResNet-50', 'resnet_101': 'ResNet-101', 'resnet_152':'ResNet-152'}

for model_name in L2_acc:
	with open(os.path.join(log_dir_cifar, '%s_new_PGD.txt'%(model_name))) as f:
	# with open(os.path.join(log_dir_cifar, '%s.txt'%(model_name.replace('alexnet', 'alexnet_linear').replace('resnet', 'resnet_nobatchnorm')))) as f:
	# with open(os.path.join(log_dir_cifar, '%s.txt'%(model_name.replace('resnet', 'resnet_nobatchnorm')))) as f:
		for line in f:
			if not line.startswith('Accuracy'):
				continue
			elif line.startswith('Accuracy on clean'):
				acc = re.search(r': (.*?)\%', line).group(1)
				cnns['clean'][model_name] = float(acc)
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

for attack in list(cnns.keys())[:-1]:
	epss = cnns[attack]['resnet_18']
	n_eps = len(epss)
	plt.clf()
	fig = plt.figure()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, cnns[attack].keys()):
		# plt.plot([0]+attack_params[attack], [cnns['clean'][model_name]]+[cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name, color=color2)
		plt.plot(np.arange(6), [cnns['clean'][model_name]]+[cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name, color=color2)
	plt.xticks(np.arange(n_eps+1), [0]+attack_params[attack])
	plt.ylim(5, 92)
	# plt.xlabel('epsilon')
	# plt.ylabel('accuracy')
	box = ax.get_position()
	plt.tight_layout()
	# ax.set_position([box.x0, box.y0,
 #                 box.width, box.height * 0.9])
	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=4)
	# plt.title('test accuracy under %s adversarial attack'%(attack))
	# plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_AlexNet_Linear_vs_ResNets.png'%(attack))
	# plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_AlexNet_vs_ResNets.png'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/%s_AlexNet_vs_ResNets_NoBN.png'%(attack))



fig = plt.figure(dpi=350, figsize=(13, 0.8))
ax = fig.add_axes([0, 0, 0.001, 0.001])
for i, model_name in enumerate(model_names):
    ax.bar(range(10), range(10), label=model_names[model_name], color=colors2[i])



plt.legend(loc="upper center", bbox_to_anchor=(500, 800), ncol=6)
plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/legend.png')


########################################################################################################################################################
### better draw AlexNet v.s. ResNet on cifar
########################################################################################################################################################


plt.clf()
plt.rc('font', family='sans-serif', serif='Times', weight='normal', size=15)
fig, axs = plt.subplots(1, 2, figsize=(9,4), dpi=200,
                                     sharey=True, tight_layout=True)

title_font = 18
plt.ylim(5, 92)
for ax, attack in zip(axs, base_cnns):
	if attack == 'L_2':
		ax.set_title(r'$L_2$ robustness', fontsize=title_font)
	else:
		ax.set_title(r'$L_{\infty}$ robustness', fontsize=title_font)
	ax.set_xlabel(r'$\epsilon$', fontsize=title_font)
	epss = cnns[attack]['resnet_18']
	n_eps = len(epss)
	for color1, color2, model_name in zip(colors1, colors2, cnns[attack].keys()):
		ax.plot(np.arange(5), [cnns['clean'][model_name]]+[cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_names[model_name], color=color2)
	if attack == 'L_2':
		ax.set_ylabel('Accuracy', fontsize=title_font)
		fig.legend(loc='upper center',
		           ncol=5, bbox_to_anchor=(0.52,1.15), columnspacing=0.5, borderaxespad=0.2, handletextpad=0.2,
		            borderpad=0.2, fontsize=15)
	else:
		ax.set_xticks(np.arange(n_eps+1))
		ax.set_xticklabels([0]+attack_params[attack], fontsize=15)



plt.savefig('/vulcanscratch/songweig/plots/adv_pool/cifar10/cifar10.png', dpi=300, bbox_inches='tight')


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

