import os
from random import shuffle
from tqdm import tqdm

num_u = 50000
num_p = 153985


def split(path, split_ratio, method, num_ratings):
	test_file = '../svdfeature/svdfeature-1.2.2/demo/basicMF/'+method+'_test.txt'
	train_file = '../svdfeature/svdfeature-1.2.2/demo/basicMF/'+method+'_train.txt'
	test_f = open(test_file, 'w')
	train_f = open(train_file, 'w')
	i = 0
	data = []
	with open(path, 'r') as file:
		for line in tqdm(file):
			data.append(line)

	shuffle(data)
	train = data[:int(num_ratings*split_ratio)]
	test = data[int(num_ratings*split_ratio):]
	for t in tqdm(train):
		train_f.write(t)
	for t in tqdm(test):
		test_f.write(t)
	test_f.close()
	train_f.close()

def run_bmf(method):
	os.chdir('../svdfeature/svdfeature-1.2.2/demo/basicMF/')
	comm = ["../../tools/make_feature_buffer {} ua.base.buffer ".format((method+'_train.txt')), "../../tools/make_feature_buffer {} ua.test.buffer ".format((method+'_test.txt'))]
	run = '../../svd_feature basicMF.conf num_round=40 '
	os.system(comm[0])
	os.system(comm[1])
	os.system(run)

if __name__ == '__main__':

	'''
	#Metapath2vec
	num_ratings = 270504
	print('Metapath2vec')
	split('metapath2vec/SVDFeature_input.txt', 0.7, 'metapath2vec', num_ratings = 137156)
	run_bmf('metapath2vec')
	os.chdir('../../../../reco')

	#Node2vec
	num_ratings = 270504
	print('Node2vec')
	split('node2vec/SVDFeature_input.txt', 0.7, 'node2vec', num_ratings = 270504)
	run_bmf('node2vec')
	'''
	#num_ratings = 270504
	#print('CBOI')
	#split('cboi/SVDFeature_input.txt', 0.7, 'cboi', num_ratings = num_ratings)
	#run_bmf('cboi')

	num_ratings = 270504
	print('SKIP')
	split('skip_gram/SVDFeature_input.txt', 0.7, 'skip_gram', num_ratings = num_ratings)
	run_bmf('skip_gram')






