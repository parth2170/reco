import pickle
import pandas as pd 
import ast
from tqdm import tqdm
import pickle

def rating():
	data = []
	with open('data/reviews.json', 'r') as file:
		for line in tqdm(file):
			try:
				jline = ast.literal_eval(line)
			except SyntaxError:
				continue
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
	with open('saved/user_codes.pickle', 'wb') as file:
		pickle.dump(user_codes, file)
	with open('saved/product_codes.pickle', 'wb') as file:
		pickle.dump(product_codes, file)
	return data


def node2vec_file():
	#Read node2vec embeddings file and make dictionary
	emb_data = {}
	with open('node2vec/', 'r') as file:
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
	with open(outfiles+"SVDFeature_input.txt", 'w') as file:
		for index, row in data.iterrows():
			try:
				i +=1
				line1 = str(int(row['overall'])) + " " + str(0) + " " + str(300) + " " + str(300) + " " + str(user_codes[row['reviewerID']]) + ":" + str(emb_data[row['reviewerID']])[1:-1].replace(", ", " " + str(user_codes[row['reviewerID']]) + ":") + " " + str(product_codes[row['asin']]) + ":" + str(emb_data[row['asin']])[1:-1].replace(", ", " "+ str(product_codes[row['asin']]) + ":") + "\n"
				file.write(line1)
			except KeyError as error:
				print(error)
	print("Number of examples in SVDFeature = {}".format(i))


def metapath2vec_file(user_codes, product_codes, data):
	emb_data = {}
	with open('metapath2vec/metapath2vec_embeddings.txt', 'r') as file:
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
	data = rating()
	with open('saved/user_codes.pickle', 'rb') as file:	
		user_codes = pickle.load(file)
	with open('saved/product_codes.pickle', 'rb') as file:
		product_codes = pickle.load(file)
	metapath2vec_file(user_codes, product_codes, data)