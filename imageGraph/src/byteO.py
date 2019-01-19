import pickle 
import numpy as np
import struct
deep_features_pth = "../deep_autoe_features.pickle"

f = open(deep_features_pth,"rb")
deep_feat = pickle.load(f)
f.close()
ain = np.load("../prod_feat_ref_list.npy")
ain = ain.tolist()
feat = np.load("../feat_list.npy")
print ("Loaded")
print ( )
f = open("feat_300.b","wb")
g = open("feat_4096.b","wb")
for index,key in enumerate(ain) :
    f.write(key.encode())
    g.write(key.encode())
    for i in deep_feat[index]:
        f.write(bytearray(struct.pack("f", i)))
    for j in feat[index]:
        g.write(bytearray(struct.pack("f", j)))
f.close()
g.close()