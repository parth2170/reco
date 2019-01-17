import numpy as np 
import json
import ast
import array
import pickle
import gc

map_dict = {}

def read_meta(meta_data_path):
	unrelated = []
	prod_cats = {}
	master_cats = ['Baby', 'Boots', 'Boys', 'Girls', 'Jwelery', 'Men', 'Novelty', 'Costumes', 'Shoes', 'Accessories', 'Women']
	print("Reading Meta Data")
	with open(meta_data_path, 'r') as file:
		count = 1
		for line in file:
			if count%20000 == 0:
				print(count)
			count += 1
			jline = ast.literal_eval(line)
			try:
				dt = dict((k, jline[k]) for k in ('asin', 'related', 'categories'))
				prod_id = dt['asin']
				cats = list(set([j for i in dt['categories'] for j in i]))
				flag = 0
				temp = []
				for cat in cats:
					if cat in master_cats:
						flag = 1
						if cat not in prod_cats:
							prod_cats[cat] = []
						prod_cats[cat].append(prod_id)
					else:
						temp.append(cat)
				for cat in temp:
					t = cat.split()
					for q in t:
						if q in master_cats:
							flag = 1
							if q not in prod_cats:
								prod_cats[q] = []
							prod_cats[q].append(prod_id)
				if flag == 0:
					print(prod_id)
			except KeyError as error:
				unrelated.append(jline['asin'])
	for cat in prod_cats:
		prod_cats[cat] = list(set(prod_cats[cat]))
	prod_cats['Novelty Costumes'] = list(set(prod_cats['Novelty'] + prod_cats['Costumes']))
	prod_cats['Shoes and Accessories'] = list(set(prod_cats['Shoes'] + prod_cats['Accessories']))
	prod_cats.pop('Novelty')
	prod_cats.pop('Costumes')
	prod_cats.pop('Shoes')
	prod_cats.pop('Accessories')
	print("Saving Dictionary")
	with open('saved/prod_cats.pickle', 'wb') as file:
		pickle.dump(prod_cats, file)
	return prod_cats, unrelated


def relation(meta_data_path, prod_cats):

	ab = open('saved/also_bought.txt', 'w')
	av = open('saved/also_viewed.txt', 'w')
	bot = open('saved/bought_together.txt', 'w')
	bav = open('saved/buy_after_viewing.txt', 'w')
	print("Creating Text files")
	count = 1
	with open(meta_data_path, 'r') as file:
		for line in file:
			if count%20000 == 0:
				print(count)
			count += 1
			jline = ast.literal_eval(line)
			dt = None
			try:
				dt = dict((k, jline[k]) for k in ('asin', 'related', 'categories'))
			except KeyError as error:
				continue

			prod_id = dt['asin']
			tcat = map_dict[prod_id]
			bt = dt['related']
			for relation in bt:
				rel_p = bt[relation]
				temp = []
				for t in rel_p:
					try:
						if map_dict[t] in tcat:
							temp.append(t)
					except KeyError:
						continue
				rel_p = temp
				line = str(prod_id) + " " + str(relation) + " " + ' '.join(rel_p) + "\n"
				if relation == 'also_bought':
					ab.write(line)
				elif relation == 'also_viewed':
					av.write(line)
				elif relation == 'bought_together':
					bot.write(line)
				elif relation == 'buy_after_viewing':
					bav.write(line)
				else:
					print(relation)
	ab.close()
	av.close()
	bot.close()
	bav.close()
	print("files saved")

def map(prod_cats):
	for cat in prod_cats:
		for prod in prod_cats[cat]:
			try:
				map_dict[prod].append(cat)
			except:
				map_dict[prod] = [cat]
				

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

def saver(cat_prod, cat_feat, n, all_feat_list, all_prod_feat_ref_list):
	for cat in cat_prod:
		np.save('saved/{}{}_prod_feat_ref_list.npy'.format(cat, str(n)), cat_prod[cat])
		np.save('saved/{}{}.npy'.format(cat, str(n)), cat_feat[cat])
	
	np.save("saved/all{}_prod_feat_ref_list.npy".format(str(n)), all_prod_feat_ref_list)
	np.save("saved/all{}.npy".format(str(n)), all_feat_list)



def image_to_npy(image_path, prod_cats):
	all_prod_feat_ref_list = []
	all_feat_list = []
	i = 1
	cat_feat = {}
	cat_prod = {}
	j = 0
	print("Reading Images")
	for image in readImageFeatures(image_path):
		if i%20000 == 0:
			print(i)
		i += 1
		im, ft = image
		im = im.decode("utf-8")
		all_prod_feat_ref_list.append(im)
		all_feat_list.append(ft)
		tcat = None
		try:
			tcat = map_dict[im]
		except KeyError:
			continue
		for cat in prod_cats:
			if cat in tcat:
				if cat not in cat_prod:
					cat_prod[cat] = []
				cat_prod[cat].append(im)
				if cat not in cat_feat:
					cat_feat[cat] = []
				cat_feat[cat].append(ft)
		if i%180000 == 0:
			print("Saving")
			saver(cat_prod, cat_feat, j, all_feat_list, all_prod_feat_ref_list)
			j+=1
			del cat_feat
			del cat_prod
			del all_feat_list
			del all_prod_feat_ref_list
			cat_feat = {}
			cat_prod = {}
			all_prod_feat_ref_list = []
			all_feat_list = []
			gc.collect()



if __name__ == '__main__':
	#pc, u =read_meta('data/meta.json')
	with open('saved/prod_cats.pickle', 'rb') as file:
		pc = pickle.load(file)
	print("Dictionary Read")
	for i in pc:
		print('{}  {}'.format(i, len(pc[i])))
	map(pc)
	#relation('data/meta.json', pc)
	image_to_npy('data/image_features', pc)



