import json
import ast
import pickle
import numpy as np 

prods_master = np.load("saved/prod_feat_ref_list.npy")

def bought_together(path, ispickle):
	#prod-prod edge occours if they have been bought together according to metadata
	#use in complete data
	i = 0
	print("\nOpening meta-data File")
	unrelated = []
	edges = []
	if ispickle:
		with open(path, 'rb') as file:
			while(True):
				try:
					jline = pickle.load(file)
					try:
						dt = dict((k, jline[k]) for k in ('asin', 'related'))
						prod_id = dt['asin']
						bt = [dt['related'][keys] for keys in dt['related'].keys()]
						bt = [j for i in bt for j in i]
						bt = list(set(bt))
						for p in bt:
							if (p in prods_master) and (prod_id in prods_master):
								edges.append((prod_id, p))
					except KeyError as error:
						unrelated.append(jline['asin'])
				except EOFError:
					break
	else:
		with open(path, 'r') as file:
			for line in file:
				if(i%10000 == 0):
					print(i)
				jline = ast.literal_eval(line)
				i+=1
				try:
					dt = dict((k, jline[k]) for k in ('asin', 'related'))
					prod_id = dt['asin']
					bt = [dt['related'][keys] for keys in dt['related'].keys()]
					bt = [j for i in bt for j in i]
					bt = list(set(bt))
					for p in bt:
						if (p in prods_master) and (prod_id in prods_master):
								edges.append((prod_id, p))
				except KeyError as error:
					unrelated.append(jline['asin'])
	print("Data read")
	print("\n++++++++++ prod-prod edges based on bought_together criteria ++++++++++\n")
	print("Number of edges = {}".format(len(edges)))
	print("Number of products with no related products = {}".format(len(unrelated)))
	return edges, unrelated


def image_similarity():
	#Implement from 
	#http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf
	#https://dl.acm.org/citation.cfm?id=2783381

	return None

 
