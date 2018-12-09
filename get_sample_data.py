import pickle 
import array
import ast
import json
import numpy as np 

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

print("Reading Images")
print("Creating sample dataset")
sample_size = 1000
feat_dict = {}
prod_feat_ref_list = []
feat_list = []
i = sample_size
for image in readImageFeatures("data/image_features"):
	if i%100 == 0:
		print(i)
	if i == 0:
		break
	im, ft = image
	im = im.decode("utf-8")
	feat_dict[im] = ft
	prod_feat_ref_list.append(im)
	feat_list.append(ft)
	i -= 1

print("Saving")
np.save("data/sample/prod_feat_ref_list.npy", prod_feat_ref_list)
np.save("data/sample/feat_list.npy", feat_list)
with open('data/sample/feat_dict.pickle', 'wb') as fp:
    pickle.dump(feat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def meta_sample():
	print("\nOpening meta-data File")
	unrelated = []
	edges = []
	i = 0
	fo = open("data/sample/meta_sample", "w")
	with open('data/meta.json', 'r') as file:
		for line in file:
			if(i%10000 == 0):
				print(i)
			jline = ast.literal_eval(line)
			if jline['asin'] in prod_feat_ref_list:
				fo.write(json.dumps(jline))
			i+=1
	fo.close()
	print("Meta Sample Created")

def rev_sample():
	i = 0
	print("\nOpening Reviews File")
	fo = open("data/sample/reviews_sample", "w")
	with open('data/reviews.json', 'r') as file:
		for line in file:
			if(i%10000 == 0):
				print(i)
			jline = json.loads(line)
			i+=1
			if jline['asin'] in prod_feat_ref_list:
				fo.write(json.dumps(jline))
	fo.close()
	print("Reviews Sample Created")

meta_sample()
rev_sample()
