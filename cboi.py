import numpy as np 
import pickle
import gc
from tqdm import tqdm

print('Loading Images')
pf = np.load('skip_gram/all.npy')
print('Loading Ids')
pid = np.load('skip_gram/all_prod_feat_ref_list.npy')
print('Making dictionary')
prod_feat = dict(zip(pid, pf))
del pf
del pid
gc.collect()
print('Loading Users')
user_feat = {}
with open('metapath2vec/user_prod_dict_mod.pickle', 'rb') as file:
	up = pickle.load(file)
c1 = 0
for user in tqdm(up):
	pfeat = []
	for prods in up[user]:
		try:
			pfeat.append(prod_feat[prods])
		except KeyError as e:
			c1 += 1
	user_feat[user] = np.average(pfeat, axis = 0)

print('Products not found = {}'.format(c1))
print('Total Products features = {}'.format(len(prod_feat)))
print('Saving')
with open('cboi/user_embed_cboi.pickle', 'wb') as file:
	pickle.dump(user_feat, file)
with open('cboi/prod_embed_cboi.pickle', 'wb') as file:
	pickle.dump(prod_feat, file)