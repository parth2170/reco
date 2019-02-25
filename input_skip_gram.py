import numpy as np 
import pickle
import math
import collections
import itertools
from tqdm import tqdm
import os
import gc

data_index = 0

class Options(object):

	def __init__(self, datafile, vocabulary_size):
		self.vocabulary_size = vocabulary_size
		self.save_path = "skip_gram"
		self.vocabulary = self.read_data(datafile)
		data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary, self.vocabulary_size)
		del self.vocabulary
		gc.collect()
		self.train_data = data_or
		self.sample_table = self.init_sample_table()

	def read_data(self, pathfile):
		print('Reading paths')
		paths = []
		with open(pathfile, 'r') as (file):
			i = 0
			for line in tqdm(file):
				paths.extend(line[:-1].split())
				i+=1
		print("Data Read")
		return paths

	def build_dataset(self, words, n_words):
		count = [['UNK', -1]]
		count.extend(collections.Counter(words).most_common(n_words - 1))
		dictionary = dict()
		print('Making dictionary')
		for word, _ in tqdm(count):
			dictionary[word] = len(dictionary)
		data = list()
		unk_count = 0
		for word in words:
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0
				unk_count += 1
			data.append(index)
		count[0][1] = unk_count
		reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		print("Saving reversed dictionary")
		with open('skip_gram/reversed_dictionary.pickle', 'wb') as file:
			pickle.dump(reversed_dictionary, file)
		return data, count, reversed_dictionary

	def init_sample_table_mod(self):
		print('Making Modified Sampling Table')
		with open('metapath2vec/user_prod_dict.pickle', 'rb') as file:
			up = pickle.load(file)
		users = list(up.keys())
		del up
		gc.collect()
		count_ = [ele[1] for ele in self.count if ele[0] in users]
		pow_freq = np.array(count_)**0.75
		power = sum(pow_freq)
		ratio = pow_freq/power
		table_size = 1e8
		count_ = np.round(ratio*table_size)
		sample_table = []
		for idx, x in tqdm(enumerate(count_)):
			sample_table += [idx] * int(x)
		return np.array(sample_table)

	def init_sample_table(self):
		print('Making Sampling Table')
		count_ = [ele[1] for ele in self.count]
		pow_freq = np.array(count_)**0.75
		power = sum(pow_freq)
		ratio = pow_freq/power
		table_size = 1e8
		count_ = np.round(ratio*table_size)
		sample_table = []
		for idx, x in tqdm(enumerate(count_)):
			sample_table += [idx] * int(x)
		return np.array(sample_table)

	def generate_batch(self, window_size, batch_size, neg_count):
		data = self.train_data 
		global data_index
		span = 2 * window_size + 1
		context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
		labels = np.ndarray(shape=(batch_size), dtype=np.int64)
		pos_pair = []
		if data_index + span > len(data):
			data_index = 0
			self.process = False
		buffer = data[data_index:data_index + span]
		pos_u = []
		pos_v = []
		for i in range(batch_size):
			data_index += 1
			context[i,:] = buffer[:window_size] + buffer[window_size + 1:]
			labels[i] = buffer[window_size]
			if data_index + span > len(data):
				buffer[:] = data[:span]
				data_index = 0
				self.process = False
			else:
				buffer = data[data_index:data_index + span]
			for j in range(span - 1):
				pos_u.append(labels[i])
				pos_v.append(context[i,j])
		neg_v = np.random.choice(self.sample_table, size = (batch_size*2*window_size, neg_count))
		return np.array(pos_u), np.array(pos_v), neg_v

