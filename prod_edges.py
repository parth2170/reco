import json
import ast
import pickle
import numpy as np 
from tqdm import tqdm
import random


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
							#if (p in prods_master) and (prod_id in prods_master):
							edges.append((prod_id, p))
					except KeyError as error:
						unrelated.append(jline['asin'])
				except EOFError:
					break
	else:
		with open(path, 'r') as file:
			for line in tqdm(file):
				jline = ast.literal_eval(line)
				i+=1
				try:
					dt = dict((k, jline[k]) for k in ('asin', 'related'))
					prod_id = dt['asin']
					bt = [dt['related'][keys] for keys in dt['related'].keys()]
					bt = [j for i in bt for j in i]
					bt = list(set(bt))
					bt = random.sample(bt, int(0.10*len(bt)))
					for p in bt:
						#if (p in prods_master) and (prod_id in prods_master):
						edges.append((prod_id, p))
				except KeyError as error:
					unrelated.append(jline['asin'])
	print("Data read")
	print("\n++++++++++ prod-prod edges ++++++++++\n")
	print("Number of edges = {}".format(len(edges)))
	print("Number of products with no related products = {}".format(len(unrelated)))
	return edges, unrelated

