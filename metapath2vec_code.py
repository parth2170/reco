import os
import datetime
import random
import pickle
import json
from joblib import Parallel, delayed
import multiprocessing
import progressbar
from tqdm import tqdm
import gc
import numpy as np 


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

def metapath_gen(user, numwalks, walklength, user_prod_dict, prod_user_dict, cold):
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
			try:
				cold[prod]
			except:
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
	size = 100
	window = 7
	negative = 5
	outpath = "../reco/"+outpath
	embout = "../"+embout+'100'
	cmd = "./metapath2vec -train "+outpath+" -output "+embout+" -pp "+str(pp)+" -size "+str(size)+" -window "+str(window)+" -negative "+str(negative)+" -threads 32"
	os.system(cmd)
	print("Embddings saved at "+embout)

def distance(code_dir, embout):
	os.chdir(code_dir)	
	cmd = "./distance "+embout
	os.system(cmd)

def reverse_dict(D):
	rD = {}
	for i in tqdm(D):
		for j in D[i]:
			try:
				rD[j].append(i)
			except KeyError:
				rD[j] = []
				rD[j].append(i)
	return rD

def mod_dict(D, min_num):
	print('Modifying Dictionary')
	new_D = {}
	for i in tqdm(D):
		if len(D[i]) > min_num:
			new_D[i] = D[i]
	rD = reverse_dict(new_D)
	#Check if products' features are available
	print('Checking if prods have image features')
	print('Loading Images')
	pf = np.load('skip_gram/all.npy')
	print('Loading Ids')
	pid = np.load('skip_gram/all_prod_feat_ref_list.npy')
	print('Making dictionary')
	prod_feat = dict(zip(pid, pf))
	prods = list(rD.keys())
	c = 0
	for i in tqdm(prods):
		try:
			prod_feat[i]
		except KeyError as e:
			rD.pop(i)
			c += 1
	print(c)
	new_D = reverse_dict(rD)
	print('Number of users = {}'.format(len(new_D)))
	#Reduce number of users
	num_users = 50000
	random.seed(777)
	sample = random.sample(new_D.keys(), num_users)
	sampled_dict = { your_key: new_D[your_key] for your_key in sample }
	rD = reverse_dict(sampled_dict)
	return sampled_dict, rD

def make_cold():
	with open('metapath2vec/prod_user_dict_mod.pickle', 'rb') as file:
	    pu = pickle.load(file)
	cold = {}
	for i in pu:
	    if len(pu[i]) >= 13:
	        cold[i] = pu[i]
	print('Number of cold Products = {}'.format(len(cold)))
	with open('metapath2vec/cold.pickle', 'wb') as file:
		pickle.dump(cold, file)
	return cold

def main():
	numwalks = 30
	walklength = 15
	reviews = "data/reviews.json"
	outpath = "metapath2vec/metapaths.txt"
	embout = "reco/metapath2vec/metapath2vec_embeddings"
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
		with open('metapath2vec/user_prod_dict.pickle', 'rb') as file:
			D = pickle.load(file)
		print('Original Number of users = {}'.format(len(D)))
		user_prod_dict, prod_user_dict = mod_dict(D = D, min_num = 2)
		del D
		gc.collect()
		print('Reduced number of users = {}'.format(len(user_prod_dict)))
		print('Reduced number of prods = {}'.format(len(prod_user_dict)))
		print('Saving New Dictionaries')
		with open('metapath2vec/user_prod_dict_mod.pickle', 'wb') as file:
			pickle.dump(user_prod_dict, file)
		with open('metapath2vec/prod_user_dict_mod.pickle', 'wb') as file:
			pickle.dump(prod_user_dict, file)
		cold = make_cold()
		results = []
		for user in tqdm(user_prod_dict):
			results.append(metapath_gen(user = user, numwalks = numwalks, walklength = walklength, user_prod_dict = user_prod_dict, prod_user_dict = prod_user_dict, cold = cold))
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
