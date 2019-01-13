import os
import datetime
import random
import pickle
from progressbar import ProgressBar 
pbar = ProgressBar()


def read_data(reviews, ispickle, min_rating):
	user_prod_dict = {}
	prod_user_dict = {}
	print("\nOpening reviews file")
	if ispickle:
		with open(reviews, 'rb') as file:
			while(True):
				try:
					jline = pickle.load(file)
					dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
					if (dt['reviewerID'] not in user_prod_dict) and (dt['overall'] > min_rating):
						user_prod_dict[dt['reviewerID']] = []
					if (dt['asin'] not in prod_user_dict) and (dt['overall'] > min_rating):
						prod_user_dict[dt['asin']] = []
					if dt['overall'] > min_rating:	#Construct user product link only if overall rating is > min_rating
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
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
				if (dt['reviewerID'] not in user_prod_dict) and (dt['overall'] > min_rating):
					user_prod_dict[dt['reviewerID']] = []
				if (dt['asin'] not in prod_user_dict) and (dt['overall'] > min_rating):
					prod_user_dict[dt['asin']] = []
				if dt['overall'] > min_rating:	#Construct user product link only if overall rating is > min_rating
					user_prod_dict[dt['reviewerID']].append(dt['asin'])
					prod_user_dict[dt['asin']].append(dt['reviewerID'])
	print("Data read")
	return user_prod_dict, prod_user_dict

def metapath_gen(user_prod_dict, prod_user_dict, numwalks, walklength, outpath):
	print("Number of users = {}".format(len(user_prod_dict)))
	print("Numver of products = {}".format(len(prod_user_dict)))
	outfile = open(outpath, 'w')
	for user in pbar(user_prod_dict):
		user0 = user
		for j in range(numwalks):
			outline = user0
			for i in range(walklength):
				prods = user_prod_dict[user]
				nump = len(prods)
				prodid = random.randrange(nump)
				prod = prods[prodid]
				outline = " "+prod
				users = prod_user_dict[prod]
				numu = len(users)
				userid = random.randrange(numu)
				user = users[userid]
				outline = " "+user
			outfile.write(outline + "\n")
	outfile.close()
	print("Metapaths saved at " + outpath)

def metapath2vec(code_dir, metapaths, embout):
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
	reviews = "saved/reviews_sample"
	outpath = "metapath2vec/metapaths.txt"
	embout = "metapath2vec/metapath2vec_embeddings.txt"
	metapath2vec_dir = "/Users/deepthought/code/docsim/code_metapath2vec"

	print("Please specify all the parameters and paths in the script itself")
	print("Enter 1 to read data and generate metapaths")
	print("Enter 2 to run metapath2vec on generated metapaths")
	print("Enter 3 to run distance on generated embeddings")
	print("Enter 4 to do all")
	task = int(input("Enter : "))
	if task == 1 or task == 4:
		numwalks = 200
		walklength = 30
		user_prod_dict, prod_user_dict = read_data(reviews = reviews, ispickle = True, min_rating = 2)		
		metapath_gen(user_prod_dict = user_prod_dict, prod_user_dict = prod_user_dict, numwalks = numwalks, walklength =  walklength, outpath = outpath)
	if task == 2 or task == 4:
		metapath2vec(code_dir = metapath2vec_dir, outpath = outpath, embout = embout)
	if task == 3 or task == 4:
		distance(code_dir, embout)
	else:
		print("Invalid Choice")

if __name__ == '__main__':
	main()
