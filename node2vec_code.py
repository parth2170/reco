import sys
import json
import pickle
import psutil
import numpy as np 
import pandas as pd 
import networkx as nx 
from node2vec import Node2Vec
from matplotlib import pyplot as plt 
from prod_edges import bought_together 

user_codes = {}
product_codes = {}
network_path = "node2vec/"
outfiles = "node2vec/"
emb_path = outfiles+"sample_node2vec_embeddings"
reviews_file = "saved/reviews_sample"


def prod_user_edge(path, ispickle, min_rating):
	data = []
	i = 0
	print("\nOpening Reviews File")
	if ispickle:
		with open(path, 'rb') as file:
			while(True):
				try:
					jline = pickle.load(file)
					dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
					data.append(dt)
				except EOFError:
					break
	else:
		with open(path, 'r') as file:
			for line in file:
				if(i%10000 == 0):
					print(i)
				jline = json.loads(line)
				i+=1
				dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
				data.append(dt)
	data = pd.DataFrame(data)
	print("Data read")
	#Product user edge exists if the product has been rated >2 by a user
	data = data.loc[data['overall'] > min_rating]
	edges = list(set(data[['reviewerID', 'asin']].itertuples(index = False)))
	print("\n+++++++++++++++ prod-user edges +++++++++++++++")
	print("Number of edges = {}".format(len(edges)))
	return edges, data

def user_user_edge(smp):
	#Two users have an edge if they puchase the same product >=smp no. of times
	return None

def prod_prod_edge():
	edges, unrelated = bought_together(path = 'data/meta.json', ispickle = False)
	return edges

def graph():
	#set parameter smp
	smp = 5
	min_rating = 2

	pue, data = prod_user_edge(path = 'data/reviews.json', ispickle = False, min_rating = min_rating)
	uue = user_user_edge(smp)
	ppe = prod_prod_edge()
	users = list(set(data['reviewerID']))

	prods = list(set(data['asin']))
	new_prods = []
	for i in ppe:
		new_prods.append(i[0])
		new_prods.append(i[1])

	new_prods = list(set(new_prods))
	prods.extend(new_prods)
	prods = list(set(prods))
	G = nx.Graph()
	G.add_nodes_from(users)
	G.add_nodes_from(prods)
	G.add_edges_from(pue)
	G.add_edges_from(ppe)

	print("\n++++++++++ Graph Information ++++++++++")
	print("Number of users = {}".format(len(users)))
	print("Number of products = {}".format(len(prods)))
	print("Number of p-u edges = {}".format(len(pue)))
	print("Number of p-p edges = {}".format(len(ppe)))
	print(nx.info(G))

	'''
	print("Visualizing Graph")
	pos=nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, nodelist = users, node_color = 'r')
	nx.draw_networkx_nodes(G, pos, nodelist = prods, node_color = 'b')
	nx.draw_networkx_edges(G,pos, edgelist = pue, width=1, alpha=0.5,edge_color='g')
	nx.draw_networkx_edges(G,pos, edgelist = new_ppe, width=8, alpha=0.5,edge_color='y')

	plt.show()
	'''

	print("Saving Graph as pickle")
	nx.write_gpickle(G, outfiles+"network.gpickle")

	#print("Saving graph edges as text file")
	#all_edges = pue + ppe
	#f = open(outfiles+'edges.txt', 'w')
	#for t in all_edges:
	#    line = ' '.join(str(x) for x in t)
	#    f.write(line + '\n')
	#f.close()

	#print("Saving users and products list as npy files")
	#np.save(outfiles+"users_node2vec.npy", users)
	#np.save(outfiles+"prods_node2vec.npy", prods)
	return G


def node2vec(graph):
	if graph == None:
		graph = nx.read_gpickle(network_path+"network.gpickle")

	node2vec = Node2Vec(graph, dimensions=300, walk_length=20, num_walks=100, workers=int(psutil.cpu_count())) 
	print("Saving paths as txt")
	with open(outfiles+"sample_paths_node2vec.txt", 'w') as foo:
		for q in node2vec.walks:
			path = ' '.join(q)
			outline = path+"\n"
			foo.write(outline)
	print("Saved")
	model = node2vec.fit(window=8, min_count=1, batch_words=5)
	model.wv.save_word2vec_format(outfiles+"sample_node2vec_embeddings")
	model.save(outfiles+"sample_model")

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
		node2vec(graph = G)
	if task == 3:
		build_svd_feat_file()

if __name__ == '__main__':
	main()
