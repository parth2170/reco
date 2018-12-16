#Gets 1000 images from the images file, finds coreresponding reviews and metadata and creates their sample
#input - raw data files
#outputs - reviews_sample, meta_sample, prod->feature dictionary, prods->feat dictionary, prods_ref_list, features_list
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
np.save("saved/prod_feat_ref_list.npy", prod_feat_ref_list)
np.save("saved/feat_list.npy", feat_list)
with open('saved/feat_dict.pickle', 'wb') as fp:
    pickle.dump(feat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def meta_sample():
	print("\nOpening meta-data File")
	unrelated = []
	edges = []
	i = 0
	fo = open("saved/meta_sample", "wb")
	with open('data/meta.json', 'r') as file:
		for line in file:
			if(i%10000 == 0):
				print(i)
			jline = ast.literal_eval(line)
			if jline['asin'] in prod_feat_ref_list:
				pickle.dump(jline, fo, protocol=pickle.HIGHEST_PROTOCOL)
			i+=1
	fo.close()
	print("Meta Sample Created")

def rev_sample():
	i = 0
	print("\nOpening Reviews File")
	fo = open("saved/reviews_sample", "wb")
	with open('data/reviews.json', 'r') as file:
		for line in file:
			if(i%10000 == 0):
				print(i)
			jline = json.loads(line)
			i+=1
			if jline['asin'] in prod_feat_ref_list:
				pickle.dump(jline, fo, protocol=pickle.HIGHEST_PROTOCOL)
	fo.close()
	print("Reviews Sample Created")

meta_sample()
rev_sample()
