
import json
import pickle
import numpy as np 
import pandas as pd 
import networkx as nx
from matplotlib import pyplot as plt 
from prod_edges import bought_together 

def read_data(data, path, ispickle):
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
				if(i%100 == 0):
					print(i)
				jline = json.loads(line)
				i+=1
				dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
				data.append(dt)
	print("Data read")
	return(pd.DataFrame(data))

def prod_user_edge(data, min_rating):
	#Product user edge exists if the product has been rated >2 by a user
	data = data.loc[data['overall'] > min_rating]
	edges = list(set(data[['reviewerID', 'asin']].itertuples(index = False)))
	print("\n+++++++++++++++ prod-user edges +++++++++++++++")
	print("Number of edges = {}".format(len(edges)))
	return edges

def user_user_edge(data, smp):
	#Two users have an edge if they puchase the same product >=smp no. of times
	return None

def prod_prod_edge():
	edges, unrelated = bought_together(path = 'data/sample/meta_sample', ispickle = True)
	return edges

#set parameter smp
smp = 5
min_rating = 2

data = read_data(data = [], path = 'data/sample/reviews_sample', ispickle = True)
pue = prod_user_edge(data, min_rating)
uue = user_user_edge(data, smp)
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

'''
comm_prods = list(set(new_prods).intersection(prods))
print("\nNumber of common products having prod-prod edges = {}".format(len(comm_prods)))
print("Keeping only these products for the testing phase")

new_ppe = []
new_prods = [] #No need to keep all the new products
#Separating out those products from prod-prod edges which are common
#Common => They have a reviewer and also a prod-prod edge
for i in ppe:
	q, w = i
	if((q in comm_prods) or (w in comm_prods)):
		new_ppe.append(i)
		new_prods.append(q)
		new_prods.append(w)

new_prods = list(set(new_prods))

print("Final number of Product-Product edges = {}".format(len(new_ppe)))
#Merging the two lists
prods = prods + new_prods
prods = list(set(prods))
'''

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
nx.write_gpickle(G, "saved/network.gpickle")

print("Saving graph edges as text file")
all_edges = pue + ppe
f = open('saved/edges.txt', 'w')
for t in all_edges:
    line = ' '.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

print("Saving users and products list as npy files")
np.save("saved/users_master.npy", users)
np.save("saved/prods_master.npy", prods)

