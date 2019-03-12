import gc
import sys
import json
import pickle
import psutil
import random
import datetime
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import networkx as nx 
from node2vec import Node2Vec
from matplotlib import pyplot as plt 
from prod_edges import bought_together
from networkx.algorithms import community 

user_codes = {}
product_codes = {}
network_path = "node2vec/"
outfiles = "node2vec/"
emb_path = outfiles+"sample_node2vec_embeddings"
reviews_file = "saved/reviews_sample"


def user_user_edge(smp):
	#Two users have an edge if they puchase the same product >=smp no. of times
	return None

def prod_prod_edge():
	temp, unrelated = bought_together(path = 'data/meta.json', ispickle = False)
	with open('metapath2vec/prod_user_dict_mod.pickle', 'rb') as file:
		prods_master = pickle.load(file)
	edges = []
	for e in tqdm(temp):
		e1, e2 = e
		try:
			prods_master[e1]
			prods_master[e2]
			edges.append(e)
		except KeyError:
			continue
	return edges

def graph():
	with open('metapath2vec/cold.pickle', 'rb') as file:
		cold = pickle.load(file)
	e = list(cold.keys())
	with open('metapath2vec/user_prod_dict_mod.pickle', 'rb') as file:
		D = pickle.load(file)

	ppe = prod_prod_edge()
	G = nx.Graph(D)
	G.add_edges_from(ppe)
	G.remove_nodes_from(e)
	print("\n++++++++++ Graph Information ++++++++++")
	print(nx.info(G))

	print("Saving Graph as pickle")
	nx.write_gpickle(G, outfiles+"network.gpickle")

	return G


def node2vec(graph, name):
	node2vec = Node2Vec(graph, dimensions=100, walk_length=15, num_walks=30, workers=int(psutil.cpu_count())) 
	print("Saving paths as txt")
	with open(outfiles+"sample_paths_node2vec{}.txt".format(name), 'w') as foo:
		for q in node2vec.walks:
			path = ' '.join(q)
			outline = path+"\n"
			foo.write(outline)
	print("Saved")
	model = node2vec.fit(window=8, min_count=1, batch_words=5)
	model.wv.save_word2vec_format(outfiles+"sample_node2vec_embeddings{}".format(name))
	#model.save(outfiles+"sample_model")

	#with open(outfiles+'sample_paths_node2vec.pickle', 'wb') as fp:
	    #pickle.dump(node2vec.walks, fp, protocol=pickle.HIGHEST_PROTOCOL)


def build_svd_feat_file():
	#Read node2vec embeddings file and make dictionary
	emb_data = {}
	with open(emb_path, 'r') as file:
		i = 0
		for line in file:
			if(not i):
				i += 1
				continue
			temp = line.split(' ', 1)			
			node = temp[0]
			emb = list(map(float, temp[1][:-1].split()))
			emb_data[node] = emb 
	print("No. of embeddings made = {}".format(len(emb_data)))
	#Read reviews data from reviews file
	data = []
	with open(reviews_file, 'rb') as file:
		while(True):
			try:
				jline = pickle.load(file)
				dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
				data.append(dt)
			except EOFError:
				break
	data = pd.DataFrame(data)
	users = set(data['reviewerID'])
	prods = set(data['asin'])
	print("No. of product reviews = {}".format(len(data)))
	print("num_users - {} num_prods - {}".format(len(users), len(prods)))
	user_codes = {item:val for val,item in enumerate(users)}
	product_codes = {item:val for val,item in enumerate(prods)}
	#Construct SVDFeature input file
	i = 0
	with open(outfiles+"SVDFeature_input.txt", 'w') as file:
		for index, row in data.iterrows():
			try:
				i +=1
				line1 = str(int(row['overall'])) + " " + str(0) + " " + str(300) + " " + str(300) + " " + str(user_codes[row['reviewerID']]) + ":" + str(emb_data[row['reviewerID']])[1:-1].replace(", ", " " + str(user_codes[row['reviewerID']]) + ":") + " " + str(product_codes[row['asin']]) + ":" + str(emb_data[row['asin']])[1:-1].replace(", ", " "+ str(product_codes[row['asin']]) + ":") + "\n"
				file.write(line1)
			except KeyError as error:
				print(error)
	print("Number of examples in SVDFeature = {}".format(i))

def main():
	print("\nEnter 1 to read data and build graph")
	print("Enter 2 to run node2vec on graph")
	print("Enter 3 to build SVDFeature file from node2vec embeddings")
	print("Enter 4 to do all")
	task = int(input("Enter : "))
	G = None
	if task == 1 or task == 4:
		G = graph()
	if task == 2 or task == 4:
		if G == None:
			G = nx.read_gpickle(network_path+"network.gpickle")
		node2vec(graph = G, name = 0)
	if task == 3:
		build_svd_feat_file()

if __name__ == '__main__':
	main()
