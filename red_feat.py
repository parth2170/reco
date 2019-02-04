import json
import numpy as np 
import gc
import os
from tqdm import tqdm
import re

def reduce(Y_path, cat):
	#Reading Y matrix
	with open(Y_path) as file:
		data = json.load(file)
		Y = np.array(data['U'])
	print('Shape of Y = {}'.format(np.shape(Y)))
	#Reading features 
	feat = np.load(cat+'.npy')
	print('Shape of old features = {}'.format(np.shape(feat[0])))
	new_feat = np.matmul(feat, Y)
	print('Shape of new features = {}'.format(np.shape(new_feat[0])))
	del feat
	del Y
	gc.collect()
	return new_feat


def combine_files(cat): 
	all_files = [f for f in os.listdir('saved/') if f.endswith('.npy')]
	all_feat = []
	all_id = []
	for file in all_files:
		temp = file.split('_')[0]
		m = re.search("\d", temp)
		catch = None
		if m:
			if file[m.start()-1] == '-':
				catch = file[:m.start()-1]
			catch = file[:m.start()]
			if catch == cat:
				feat = np.load("saved/"+file)
				if len(feat[0]) == 4096:
					all_feat.extend(feat)
				else:
					all_id.extend(feat)
				del feat
				gc.collect()
	np.save('imageGraph/data/'+cat+'_prod_feat_ref_list.npy', all_id)
	np.save('imageGraph/data/'+cat+'.npy', all_feat)
	print(cat)
	print("Number of products = {}".format(len(all_id)))
	del all_feat
	del all_id
	gc.collect()

def run_paper(cat):
	cmd = 'python imageGraph/main_ex_encoder.py --img_data="data/{}.npy" --index_data="data/{}_prod_feat_ref_list.npy"'.format(cat, cat)
	os.system(cmd)

def get_relations():
	ab = {}
	with open('saved/also_bought.txt') as file:
		for line in file:
			x, y = line.split(' also_bought ')
			ab[x] = y.split()
	av = {}
	with open('saved/also_viewed.txt') as file:
		for line in file:
			x, y = line.split(' also_viewed ')
			av[x] = y.split()
	bt = {}
	with open('saved/bought_together.txt') as file:
		for line in file:
			x, y = line.split(' bought_together ')
			bt[x] = y.split()
	bv = {}
	with open('saved/buy_after_viewing.txt') as file:
		for line in file:
			x, y = line.split(' buy_after_viewing ')
			bv[x] = y.split()
	all_keys = list(set(list(ab.keys()) + list(av.keys()) + list(bt.keys()) + list(bv.keys())))
	i = 0
	with open('saved/all_relationships.txt', 'w') as file:
		for key in tqdm(all_keys):
			outline = key + ' also_viewed '
			p1, p2, p3, p4 = [], [], [], []
			try:
				p1 = ab[key]
			except KeyError:
				i += 1
			try:
				p2 = av[key]
			except KeyError:
				i += 1
			try:
				p3 = bt[key]
			except KeyError:
				i += 1
			try:
				p4 = bv[key]
			except KeyError:
				i += 1
			prods = list(set(p1 + p2 + p3 + p4))
			outline = outline + ' '.join(prods) + '\n'
			file.write(outline)


if __name__ == '__main__':

	get_relations()
	master_cats = ['Baby', 'Boots', 'Boys', 'Girls', 'Jewelry', 'Men', 'Novelty Costumes', 'Shoes and Accessories', 'Women']
	all_feat = []

	'''
	for cat in master_cats:
		try:
			#Women, Shoes, Jewelry running out of memory
			if cat == 'Women':
				print(cat)
				combine_files(cat = cat)
				#run_paper(cat = cat)
				#feat = reduce('imageGraph/Y.txt-also_viewed-100-0.000000.txt', 'saved/{}'.format(cat))
				#all_feat.extend(feat)
	'''
	combine_files(cat = 'all')

		except FileNotFoundError as error:
			print(error)
