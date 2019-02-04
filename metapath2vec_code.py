import os
import datetime
import random
import pickle
import json
from joblib import Parallel, delayed
import multiprocessing
import progressbar
from tqdm import tqdm


numwalks = 100
walklength = 20

with open('metapath2vec/user_prod_dict.pickle', 'rb') as file:
	user_prod_dict = pickle.load(file)
with open('metapath2vec/prod_user_dict.pickle', 'rb') as file:
	prod_user_dict = pickle.load(file)

def read_data(reviews, ispickle, min_rating):

	user_prod_dict = {}
	prod_user_dict = {}
	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
	print("\nOpening reviews file")
	i = 0
	if ispickle:
		with open(reviews, 'rb') as file:
			while(True):
				try:
					jline = pickle.load(file)
					dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
					if (dt['reviewerID'] not in user_prod_dict) and (dt['overall'] >= min_rating):
						user_prod_dict[dt['reviewerID']] = []
					if (dt['asin'] not in prod_user_dict) and (dt['overall'] >= min_rating):
						prod_user_dict[dt['asin']] = []
					if dt['overall'] > min_rating:	#Construct user product link only if overall rating is > min_rating
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
				except EOFError:
					break
	else:
		with open(reviews, 'r') as file:
			for line in file:
				bar.update(i)
				jline = json.loads(line)
				i+=1
				dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
				if dt['overall'] > min_rating:
					try:
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
					except KeyError:
						user_prod_dict[dt['reviewerID']] = []
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
					try:
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
					except KeyError:
						prod_user_dict[dt['asin']] = []
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
	print("\nData read")
	return user_prod_dict, prod_user_dict

def metapath_gen(user):
	outfile = []
	user0 = user
	for j in range(numwalks):
		outline = user0
		for i in range(walklength):
			try:
				prods = user_prod_dict[user]
			except KeyError as error:
				print(error)
			nump = len(prods)
			if nump == 0:
				continue
			prodid = random.randrange(nump)
			prod = prods[prodid]
			outline = outline+" "+prod
			try:
				users = prod_user_dict[prod]
			except KeyError as error:
				print(error)
			numu = len(users)
			if numu == 0:
				continue
			userid = random.randrange(numu)
			user = users[userid]
			outline = outline+" "+user
		outfile.append(str(outline + "\n"))
	return outfile


def metapath2vec(code_dir, outpath, embout):
	print("Running Metapath2Vec")
	os.chdir(code_dir)
	pp = 1
	size = 128
	window = 7
	negative = 5
	outpath = "../"+outpath
	embout = "../"+embout
	cmd = "./metapath2vec -train "+outpath+" -output "+embout+" -pp "+str(pp)+" -size "+str(size)+" -window "+str(window)+" -negative "+str(negative)+" -threads 32"
	os.system(cmd)
	print("Embddings saved at "+embout)

def distance(code_dir, embout):
	os.chdir(code_dir)	
	cmd = "./distance "+embout
	os.system(cmd)

def main():

	numwalks = 50
	walklength = 20
	global user_prod_dict
	global prod_user_dict
	reviews = "data/reviews.json"
	outpath = "reco/metapath2vec/metapaths.txt"
	embout = "reco/metapath2vec/metapath2vec_embeddings.txt"
	metapath2vec_dir = "../code_metapath2vec"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 1 to read data and generate metapaths")
	print("Enter 2 to generate metapaths")
	print("Enter 3 to run metapath2vec on generated metapaths")
	print("Enter 4 to run distance on generated embeddings")
	task = int(input("Enter : "))
	if task == 1:
		user_prod_dict, prod_user_dict = read_data(reviews = reviews, ispickle = False, min_rating = 1)	
		print('Saving Dictionaries')
		with open('metapath2vec/user_prod_dict.pickle', 'wb') as file:
			pickle.dump(user_prod_dict, file)
		with open('metapath2vec/prod_user_dict.pickle', 'wb') as file:
			pickle.dump(prod_user_dict, file)
	if task == 2:
		#num_cores = multiprocessing.cpu_count()
		#print('Running on {} cores'.format(num_cores))
		#results = Parallel(n_jobs=num_cores)(delayed(metapath_gen)(user = i) for i in tqdm(user_prod_dict))	
		results = []
		for user in tqdm(user_prod_dict):
			results.append(metapath_gen(user))
		print("Saving Metapaths at " + outpath)
		with open(outpath, 'w') as file:
			for paths in tqdm(results):
				for path in paths:
					file.write(path)
	if task == 3:
		metapath2vec(code_dir = metapath2vec_dir, outpath = outpath, embout = embout)
	if task == 4:
		distance(code_dir, embout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()
