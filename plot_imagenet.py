import os
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

log_dir_imagenet = '/vulcanscratch/songweig/logs/adv_pool/imagenet_unnorm'
model_names = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
				'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 
				'wide_resnet50_2', 'densenet121', 'densenet169', 'densenet201', 'mobilenet_v2']

L2_acc = {model_name:{'0.55':0., '1.09':0., '2.18':0., '4.37':0.} for model_name in model_names}
Linf_acc = {model_name:{'0.01':0., '0.02':0., '0.03':0., '0.07':0.} for model_name in model_names}
strength_norm = {'L_2': {'0.55': '0.125', '1.09': '0.25', '2.18': '0.5', '4.37': '1'}, 'L_inf': {'0.01': '0.5/255', '0.02': '1/255', '0.03': '2/255', '0.07': '4/255'}}

base_cnns = {'L_2': copy.deepcopy(L2_acc), 'L_inf': copy.deepcopy(Linf_acc)}

clean_accs = {'alexnet': [56.55, 56.94], 'vgg11': [69.02, 70.51], 'vgg13': [69.93, 71.52], 'vgg16': [71.59, 72.96], 'vgg19': [72.38, 73.54], 
			'vgg11_bn': [70.38, 72.63], 'vgg13_bn': [71.55, 73.61], 'vgg16_bn': [73.36, 75.13], 'vgg19_bn': [74.24, 75.68], 'resnet18': [69.74, 71.67], 
			'resnet34': [73.30, 74.60], 'resnet50': [76.16, 77.41], 'resnet101': [77.37, 78.38], 'resnet152': [78.31, 79.07], 'resnext50_32x4d': [77.62, 77.93], 
			'resnext101_32x8d': [79.31, 79.33], 'wide_resnet50_2': [78.47, 78.70], 'wide_resnet101_2': [78.85, 78.99], 'densenet121': [74.43, 75.79], 
			'densenet169': [75.60, 76.73], 'densenet201': [76.90, 77.31], 'densenet161': [77.14, 77.88], 'mobilenet_v2': [71.88, 72.72]}

consistency = {'alexnet': 78.18, 'vgg11': 86.58, 'vgg13': 86.92, 'vgg16': 88.52, 'vgg19': 89.17, 'vgg11_bn': 87.16, 'vgg13_bn': 88.03, 
				'vgg16_bn': 89.24, 'vgg19_bn': 89.59, 'resnet18': 85.11, 'resnet34': 87.56, 'resnet50': 89.20, 'resnet101': 89.81, 'resnet152': 90.92, 
				'resnext50_32x4d': 90.17, 'resnext101_32x8d': 91.33, 'wide_resnet50_2': 90.77, 'wide_resnet101_2': 90.93, 'densenet121': 88.81, 
				'densenet169': 89.68, 'densenet201': 90.36, 'densenet161': 90.82, 'mobilenet_v2': 86.50}

for model_name in model_names:
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


########################################################################################################################################################
### draw accuracy under different attack v.s. clean accuracy
########################################################################################################################################################

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
		'#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
		'#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000']

font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

# y_lims = {'L_2': [6, 60], 'L_inf': [5, 22]}
y_lims = {'L_2': [0., 1.], 'L_inf': [0., 1.]}
for attack in base_cnns:
	for strength in base_cnns[attack]['alexnet']:
		plt.clf()
		fig = plt.figure()
		for i, model_name in enumerate(model_names):
			# plt.scatter([clean_accs[model_name][0]], [base_cnns[attack][model_name][strength]], color=colors[i])
			# plt.scatter([consistency[model_name]], [base_cnns[attack][model_name][strength]], color=colors[i])
			rate = (clean_accs[model_name][0]-base_cnns[attack][model_name][strength])/clean_accs[model_name][0]
			plt.scatter([consistency[model_name]], [rate], color=colors[i], s=100, alpha=0.8)
		# plt.xticks(np.arange(n_models), L2_acc.keys())
		# plt.xlabel('clean accuracy')
		# plt.ylabel('Adversarial attack')
		plt.xlabel('Consistency')
		plt.ylabel('Success rate')
		plt.ylim(*y_lims[attack])
		# plt.legend(loc='upper right')
		# plt.title('test accuracy under %s attack with epsilon %s'%(attack, strength_norm[attack][strength]))
		# plt.title('%s attack with epsilon %s'%(attack, strength_norm[attack][strength]))
		plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/pretrained_%s_%s_consistency.png'%(attack, strength))



fig = plt.figure(dpi=350, figsize=(12, 1.5))
ax = fig.add_axes([0, 0, 0.001, 0.001])
for i in range(len(model_names)):
    ax.bar(range(10), range(10), label=model_names[i], color=colors[i])



plt.legend(loc="upper center", bbox_to_anchor=(500, 800), ncol=7)
plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/legend.png')

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
	epss = base_cnns[attack]['vgg11']
	n_eps = len(epss)
	plt.clf()
	ax = plt.subplot(111)
	for color1, color2, model_name in zip(colors1, colors2, L2_acc.keys()):
		plt.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name, color=color2)
		plt.plot(np.arange(n_eps+1), [clean_accs[model_name+'_bn'][0]]+[antialiased_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name+'_bn', color=color1)
	plt.xticks(np.arange(n_eps+1), [0]+list(epss.keys()))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
	plt.title('test accuracy under %s adversarial attack'%(attack))
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/vgg_%s.png'%(attack))


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
CAND_COLORS = ['#e31a1c','#9B870C', '#1f78b4','#a6cee3','#33a02c','#b2df8a', '#ff7f00','#6a3d9a', '#61a0a8', '#c23531']

plt.rc('font', family='serif', serif='Times', size=18)
model_name_selected = {'alexnet':'AlexNet', 'resnet50':'ResNet-50', 'vgg11':'VGG-11', 'vgg11_bn':'VGG-11+BN', 'vgg19':'VGG-19', 'vgg19_bn':'VGG-19+BN', 'densenet121':'DenseNet-121', 'mobilenet_v2':'MobileNet V2'}
strength_norm = {'L_2': {'0.55': '0.125', '1.09': '0.25', '2.18': '0.5', '4.37': '1'}, 'L_inf': {'0.01': '0.5/255', '0.02': '1/255', '0.03': '2/255', '0.07': '4/255'}}
# colors = ['#003f5c', '#ffa600', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', 'green']
for attack in base_cnns:
	fig = plt.figure()
	epss = base_cnns[attack]['alexnet']
	n_eps = len(epss)
	plt.clf()
	for color, model_name in zip(CAND_COLORS, model_name_selected.keys()):
		# plt.scatter(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], label=model_name, color=color)
		if 'BN' in model_name_selected[model_name]:
			plt.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], linestyle='dashed', marker='o', label=model_name_selected[model_name], color=color)
		else:
			plt.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name_selected[model_name], color=color)
	plt.xticks(np.arange(n_eps+1), [0]+[strength_norm[attack][eps] for eps in list(epss.keys())])
	plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/%s_models.png'%(attack))



fig = plt.figure(dpi=350, figsize=(9.5, 1))
ax = fig.add_axes([0, 0, 0.001, 0.001])
for color, model_name in zip(CAND_COLORS, model_name_selected.keys()):
	if 'BN' in model_name_selected[model_name]:
		ax.plot(range(10), range(10), linestyle='dashed', marker='o', label=model_name_selected[model_name], color=color)
	else:
		ax.plot(range(10), range(10), marker='o', label=model_name_selected[model_name], color=color)



plt.legend(loc="upper center", bbox_to_anchor=(500, 800), ncol=4)
plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/legend.png')



########################################################################################################################################################
### better draw
########################################################################################################################################################

plt.clf()
plt.rc('font', family='sans-serif', serif='Times', weight='normal', size=15)
fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi=200,
                                     sharey=True, tight_layout=True)

CAND_COLORS = ['#e31a1c','#9B870C', '#1f78b4','#a6cee3','#33a02c','#b2df8a', '#ff7f00','#6a3d9a', '#61a0a8', '#c23531']

model_name_selected = {'alexnet':'AlexNet', 'resnet50':'ResNet-50', 'vgg11':'VGG-11', 'vgg11_bn':'VGG-11+BN', 'vgg19':'VGG-19', 'vgg19_bn':'VGG-19+BN', 'densenet121':'DenseNet-121', 'mobilenet_v2':'MobileNet V2'}
strength_norm = {'L_2': {'0.55': '0.125', '1.09': '0.25', '2.18': '0.5', '4.37': '1'}, 'L_inf': {'0.01': '0.5/255', '0.02': '1/255', '0.03': '2/255', '0.07': '4/255'}}
# colors = ['#003f5c', '#ffa600', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', 'green']
title_font = 18
plt.ylim(0, 80)
for ax, attack in zip(axs, base_cnns):
	if attack == 'L_2':
		ax.set_title(r'$L_2$ robustness', fontsize=title_font)
	else:
		ax.set_title(r'$L_{\infty}$ robustness', fontsize=title_font)
	ax.set_xlabel(r'$\epsilon$', fontsize=title_font)
	epss = base_cnns[attack]['alexnet']
	n_eps = len(epss)
	for color, model_name in zip(CAND_COLORS, model_name_selected.keys()):
		# plt.scatter(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], label=model_name, color=color)
		if 'BN' in model_name_selected[model_name]:
			ax.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], linestyle='dashed', marker='o', label=model_name_selected[model_name], color=color)
		else:
			ax.plot(np.arange(n_eps+1), [clean_accs[model_name][0]]+[base_cnns[attack][model_name][strength] for strength in epss], marker='o', label=model_name_selected[model_name], color=color)
	# ax.set_xticks(np.arange(n_eps+1), [0]+[strength_norm[attack][eps] for eps in list(epss.keys())])
	ax.set_xticks(np.arange(n_eps+1))
	ax.set_xticklabels([0]+[strength_norm[attack][eps] for eps in list(epss.keys())])
	if attack == 'L_2':
		ax.set_ylabel('Accuracy', fontsize=title_font)
		fig.legend(loc='upper center',
		           ncol=4, bbox_to_anchor=(0.52,1.15), columnspacing=0.5, borderaxespad=0.2, handletextpad=0.2,
		            borderpad=0.2, fontsize=15)


# fm = matplotlib.font_manager.json_load("/cfarhomes/songweig/.cache/matplotlib/fontlist-v330.json")
# fm.findfont("serif", rebuild_if_missing=False)
# fm.findfont("serif", fontext="afm", rebuild_if_missing=False)

plt.savefig('/vulcanscratch/songweig/plots/adv_pool/imagenet/imagenet.png', dpi=300, bbox_inches='tight')
