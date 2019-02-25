import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from input_skip_gram import *
from tqdm import tqdm
import gc

feat_path = 'skip_gram/all.npy'
rev_dict_path = 'skip_gram/reversed_dictionary.pickle'
pord_list_path = 'skip_gram/all_prod_feat_ref_list.npy'

class skipgram(nn.Module):

	def __init__(self, vocab_size, embedding_dim):
		super(skipgram, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.embedding_dim = embedding_dim
		self.prod_codes = []
		self.reversed_dictionary = None
		with open(rev_dict_path, 'rb') as file:
			self.reversed_dictionary = pickle.load(file)
		self.init_emb()

	def init_emb(self):
		initrange = 0.5 / self.embedding_dim
		#Make weight matrix
		print('Loading Embedding data')
		feat = np.load(feat_path)
		plist = np.load(pord_list_path).tolist()
		pord_list = dict(zip(plist, [i for i in range(len(plist))]))
		del plist
		gc.collect()
		print('Initializing Embeddings')
		emb = np.random.uniform(-initrange, initrange, size = (self.vocab_size, self.embedding_dim))
		for node in tqdm(self.reversed_dictionary):
			try:
				emb[node] = feat[pord_list[self.reversed_dictionary[node]]]
				self.prod_codes.append(node)
			except KeyError:
				continue
		emb = emb.astype(np.float64)
		self.u_embeddings.weight.data = torch.tensor(emb, dtype = torch.double)
		self.v_embeddings.weight.data.uniform_(-0,0)

	def forward(self, u_pos, v_pos, v_neg, batch_size):
		embed_u = self.u_embeddings(u_pos)
		embed_v = self.v_embeddings(v_pos)
		score = torch.mul(embed_u, embed_v)
		score = torch.sum(score, dim = 1)
		log_target = F.logsigmoid(score).squeeze()
		neg_embed_v = self.v_embeddings(v_neg)
		neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
		neg_score = torch.sum(neg_score, dim = 1)
		sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()
		loss = log_target + sum_log_sampled
		return -1*loss.sum()/batch_size

	def input_embeddings(self):
		return self.u_embeddings.weight.data.cpu().numpy()

	def save_embedding(self, file_name):
		u = []
		f = []
		with open('metapath2vec/user_prod_dict_mod.pickle', 'rb') as file:
			up = pickle.load(file)
		for idx in tqdm(self.reversed_dictionary):
			try:
				u.append(up[self.reversed_dictionary[idx]])
				f.append(self.u_embeddings.weight.data[idx])
				print(len(f))
				break
			except KeyError:
				continue
		del up
		gc.collect()
		print('Saving')
		np.save(file_name, f)
		np.save(file_name+'ref', u)
		
