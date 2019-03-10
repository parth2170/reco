import pickle
from scipy import spatial
import operator
import numpy as np
from tqdm import tqdm

def reverse_dict(D):
	return {v: k for k, v in D.items()}

def make_cold():
	with open('metapath2vec/prod_user_dict_mod.pickle', 'rb') as file:
	    pu = pickle.load(file)
	cold = {}
	for i in pu:
	    if len(pu[i]) >= 13:
	        cold[i] = pu[i]
	print('Number of cold Products = {}'.format(len(cold)))
	return cold

def rank(prod, uf, pfc):
    d0 = {}
    for user in uf:
        temp = spatial.distance.cosine(prod, uf[user])
        if temp < 0:
            temp *= -1
        d0[user] = temp
    sorted_d0 = sorted(d0.items(), key=operator.itemgetter(1))
    d0 = {sorted_d0[i][0]:i+1 for i in range(len(sorted_d0))}
    return d0

def RR(cold, prod, ranks):
	r = [ranks[user] for user in cold[prod]]
	return 1/np.min(r)

def Recall(cold, prod, ranks):
	r = [ranks[user] for user in cold[prod]]
	R = [1 if ranks[user] < len(cold[prod]) else 0 for user in cold[prod]]
	recall = np.sum(np.array(R))/len(cold[prod])
	return recall

def P_at_M(cold, prod, rev_ranks, M):
	P = 0
	for i in range(M):
		if rev_ranks[i+1] in cold[prod]:
			P += 1
	return P/M

def Results(cold, pfc, uf):
	MRR = []
	recall = []
	p5 = []
	p10 = []
	check = 5
	for prod in tqdm(cold):
		ranks = rank(pfc[prod], uf, pfc)
		MRR.append(RR(cold, prod, ranks))
		recall.append(Recall(cold, prod, ranks))
		rev_ranks = reverse_dict(ranks)
		p5.append(P_at_M(cold, prod, rev_ranks, 5))
		p10.append(P_at_M(cold, prod, rev_ranks, 10))
		if check == 0:
			break
		check -= 1
	print('MRR = {}'.format(np.average(np.array(MRR))))
	print('recall = {}'.format(np.average(np.array(recall))))
	print('p@5 = {}'.format(np.average(np.array(p5))))
	print('p@10 = {}'.format(np.average(np.array(p10))))

if __name__ == '__main__':
	cold = make_cold()
	with open('cboi/user_embed_cboi.pickle', 'rb') as file:
	    uf_cboi = pickle.load(file)
	pf = {}
	uf = {}
	print("Enter 1 for CBOI")
	print("Enter 2 for metapath2vec")
	print("Enter 3 for node2vec")
	print("Enter 4 for skip_gram")
	task = int(input("Enter : "))
	if task == 1:
		print('CBOI')
		print('Loding Data\n')
		with open('cboi/prod_embed_cboi.pickle', 'rb') as file:
			pf = pickle.load(file)
		pfc = dict((k, pf[k]) for k in list(cold.keys()))
		Results(cold, pfc, uf_cboi)
	elif task == 2:
		print('metapath2vec')
		print('Loading Data\n')
		with open('metapath2vec/metapath2vec_embeddings.txt', 'r') as (file):
			i = 0
			for line in tqdm(file):
				if i <= 1:
					i += 1
					continue
				temp = line.split(' ', 1)			
				node = temp[0]
				emb = list(map(float, temp[1][:-1].split()))
				try:
					uf_cboi[node]
					uf[node] = emb
				except:
					pf[node] = emb
		pfc = dict((k, pf[k]) for k in list(cold.keys()))
		Results(cold, pfc, uf)
	elif task == 3:
		print('node2vec')
		with open('node2vec/sample_node2vec_embeddings0', 'r') as file:
			i = 0
			for line in file:
				if(not i):
					i += 1
					continue
				temp = line.split(' ', 1)			
				node = temp[0]
				emb = list(map(float, temp[1][:-1].split()))
				try:
					uf_cboi[node]
					uf[node] = emb
				except:
					pf[node] = emb
		pfc = dict((k, pf[k]) for k in list(cold.keys()))
		Results(cold, pfc, uf)
	elif task == 4:
		print('skip_gram')
		with open('skip_gram/tmp/embeddings.epoch4.batch800000.txt', 'r') as file:
			for line in file:
				temp = line.split(' ', 1)			
				node = temp[0]
				emb = list(map(float, temp[1][:-1].split()))
				uf[node] = emb
		with open('cboi/prod_embed_cboi.pickle', 'rb') as file:
			pf = pickle.load(file)
		pfc = dict((k, pf[k]) for k in list(cold.keys()))
		Results(cold, pfc, uf)
	else:
		print('Invalid')











