import json
import ast

def bought_together():
	#prod-prod edge occours if they have been bought together according to metadata
	i = 0
	print("\nOpening meta-data File")
	unrelated = []
	edges = []
	with open('data/meta.json', 'r') as file:
		for line in file:
			if(i%100 == 0):
				print(i)
			if(i==1000):
				break
			jline = ast.literal_eval(line)
			i+=1
			try:
				dt = dict((k, jline[k]) for k in ('asin', 'related'))
				prod_id = dt['asin']
				bt = dt['related']['bought_together']
				for p in bt:
					edges.append((prod_id, p))
			except KeyError as error:
				unrelated.append(jline['asin'])
	print("Data read")
	print("\n++++++++++ prod-prod edges based on bought_together criteria ++++++++++\n")
	print("Number of edges = {}".format(len(edges)))
	print("Number pf products with no related products = {}".format(len(unrelated)))
	return edges, unrelated


def image_similarity():
	#Implement from 
	#http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf
	#https://dl.acm.org/citation.cfm?id=2783381
	return None

 
