from tqdm import tqdm
import numpy as np
import pickle 
import array


def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

prods = np.load("saved/prods_master.npy")
print("Reading Images")
feat_dict = {}
prod_feat_ref_list = []
feat_list = []
for image in readImageFeatures("data/image_features"):
	if i%10000 == 0:
		print(i)
	if i == 0:
		break
	im, ft = image
	im = im.decode("utf-8")
	if im in prods:
		feat_dict[im] = ft
		prod_feat_ref_list.append(im)
		feat_list.append(ft)

print("Saving")
np.save("saved/prod_feat_ref_list.npy", prod_feat_ref_list)
np.save("saved/feat_list.npy", feat_list)
with open('saved/feat_dict.pickle', 'wb') as fp:
    pickle.dump(feat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


