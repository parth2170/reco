import os
from random import shuffle

num_u = 50000
num_p = 153985
num_ratings = 270504

def split(path, split_ratio, method):
	test_file = '../svdfeature/svdfeature-1.2.2/demo/basicMF/'+method+'_test.txt'
	train_file = '../svdfeature/svdfeature-1.2.2/demo/basicMF/'+method+'_train.txt'
	test_f = open(test_file, 'w')
	train_f = open(train_file, 'w')
	i = 0
	data = []
	with open(path, 'r') as file:
		for line in file:
			data.append(line)

	shuffle(data)
	train = data[:int(num_ratings*split_ratio)]
	test = data[int(num_ratings*split_ratio):]
	for t in train:
		train_f.write(t)
	for t in test:
		test_f.write(t)
	test_f.close()
	train_f.close()

def run_bmf(method):
	os.chdir('../svdfeature/svdfeature-1.2.2/demo/basicMF/')
	comm = ["../../tools/make_feature_buffer {} {}.buffer ".format((method+'_train.txt'), (method+'_train')), "../../tools/make_feature_buffer {} {}.buffer ".format((method+'_test.txt'), (method+'_test'))]
	run = '../../svd_feature basicMF.conf num_round=40 '
	os.system(comm[0])
	os.system(comm[1])
	os.system(run)

if __name__ == '__main__':
	#Node2vec
	split('node2vec/SVDFeature_input.txt', 0.7, 'node2vec')
	run_bmf('node2vec')





