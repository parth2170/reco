import collections
import numpy as np

import math
import random


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time

from input_skip_gram import Options
from skip_gram_mod import skipgram

class word2vec:
	def __init__(self, inputfile, vocabulary_size = 500, embedding_dim = 300, epoch_num = 10, batch_size = 16, window_size = 5, neg_sample_num = 10):
		self.op = Options(inputfile, vocabulary_size)
		self.embedding_dim = embedding_dim
		self.window_size = window_size
		self.vocabulary_size = vocabulary_size
		self.batch_size = batch_size
		self.epoch_num = epoch_num
		self.neg_sample_num = neg_sample_num

	def train(self):
		model = skipgram(self.vocabulary_size, self.embedding_dim)
		if torch.cuda.is_available():
			model.cuda()

		optimizer = optim.SGD(model.parameters(), lr = 0.2)

		for epoch in range(self.epoch_num):

			start = time.time()
			self.op.process = True
			batch_num = 0
			batch_new = 0

			while self.op.process:

				pos_u, pos_v, neg_v = self.op.generate_batch(self.window_size, self.batch_size, self.neg_sample_num)
				pos_u = Variable(torch.LongTensor(pos_u))
				pos_v = Variable(torch.LongTensor(pos_v))
				neg_v = Variable(torch.LongTensor(neg_v))

				if torch.cuda.is_available():
					pos_u = pos_u.cuda()
					pos_v = pos_v.cuda()
					neg_v = neg_v.cuda()

				optimizer.zero_grad()
				model = model.double()
				loss = model(pos_u, pos_v, neg_v, self.batch_size)
				loss.backward()

				#Kill gradients of products
				embed_grad = next(next(model.children()).parameters()).grad
				for i in range(len(pos_u)):
					if int(pos_u[i]) in model.prod_codes:
						embed_grad[i] = torch.tensor(np.zeros(300), dtype = torch.double)

				optimizer.step()

				if batch_num % 60 == 0:
					print("Saving model")
					torch.save(model.state_dict(), 'skip_gram/tmp/skipgram.epoch{}.batch{}'.format(epoch,batch_num))
					print("Saving Embeddings")
					model.save_embedding('skip_gram/tmp/embeddings.epoch{}.batch{}.pickle'.format(epoch,batch_num))
				if batch_num%10 == 0:
					end = time.time()
					word_embeddings = model.input_embeddings()
					print('epoch = {} batch = {} loss = {} time = {}'.format(epoch, batch_num, loss, end - start))
					batch_new = batch_num
					start = time.time()
				batch_num += 1
			print("/nOptimization Finished")

if __name__ == '__main__':
	wc = word2vec('node2vec/sample_paths_node2vec.txt')
	wc.train()
