import pickle
import pandas as pd 
import ast
from tqdm import tqdm
import pickle

def rating():
	data = []
	with open('data/reviews.json', 'rb') as file:
		i = 0
		for line in tqdm(file):
			#jline = ast.literal_eval(line.decode('UTF-8'))
			i += 1
			try:
				#jline = ast.literal_eval(line)
				jline = ast.literal_eval(line.decode('UTF-8'))
			except SyntaxError:
				print(line.decode('UTF-8'))
			try:
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
	return user_codes, product_codes, data


def node2vec_file(user_codes, product_codes, data):
	#Read node2vec embeddings file and make dictionary
	emb_data = {}
	with open('node2vec/sample_node2vec_embeddings0', 'r') as file:
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

	#Construct SVDFeature input file
	
	i = 0
	with open("node2vec/SVDFeature_input.txt", 'w') as file:
		for index, row in data.iterrows():
			try:
				i +=1
				line1 = str(int(row['overall'])) + " " + str(0) + " " + str(300) + " " + str(300) + " " + str(user_codes[row['reviewerID']]) + ":" + str(emb_data[row['reviewerID']])[1:-1].replace(", ", " " + str(user_codes[row['reviewerID']]) + ":") + " " + str(product_codes[row['asin']]) + ":" + str(emb_data[row['asin']])[1:-1].replace(", ", " "+ str(product_codes[row['asin']]) + ":") + "\n"
				file.write(line1)
			except KeyError as error:
				continue
	print("Number of examples in SVDFeature = {}".format(i))


def metapath2vec_file(user_codes, product_codes, data):
	emb_data = {}
	with open('metapath2vec/metapath2vec_embeddings100.txt', 'r') as file:
		i = 0
		for line in file:
			if i <= 1:
				i += 1
				continue
			temp = line.split(' ', 1)			
			node = temp[0]
			emb = list(map(float, temp[1][:-1].split()))
			emb_data[node] = emb 
	print("No. of embeddings made = {}".format(len(emb_data)))

	#Construct SVDFeature input file
	
	i = 0
	with open("metapath2vec/SVDFeature_input.txt", 'w') as file:
		for index, row in tqdm(data.iterrows()):
			try:
				i +=1
				line1 = str(int(row['overall'])) + " " + str(0) + " " + str(300) + " " + str(300) + " " + str(user_codes[row['reviewerID']]) + ":" + str(emb_data[row['reviewerID']])[1:-1].replace(", ", " " + str(user_codes[row['reviewerID']]) + ":") + " " + str(product_codes[row['asin']]) + ":" + str(emb_data[row['asin']])[1:-1].replace(", ", " "+ str(product_codes[row['asin']]) + ":") + "\n"
				file.write(line1)
			except KeyError as error:
				continue
	print("Number of examples in SVDFeature = {}".format(i))

if __name__ == '__main__':
	user_codes, product_codes, data = rating()
	metapath2vec_file(user_codes, product_codes, data)
	node2vec_file(user_codes, product_codes, data)
