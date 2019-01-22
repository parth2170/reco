import os
from optparse import OptionParser
import pickle
import struct
from image_autoencoder import autoencode
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re

res = open("data/results.txt","w")
parser = OptionParser()
parser.add_option("--img_data", dest="img_data")
parser.add_option("--index_data", dest="index_data")
parser.add_option("--encoding_dim", dest="encoding_dim",default = 300)
parser.add_option("--optimizer", dest="optimizer",default = "adadelta")
parser.add_option("--eval_for_original", dest="eval_for_original",default = True)
parser.add_option("--plot_file",dest="plot_file_name",default = "acc.png")
(options, args) = parser.parse_args()
print (options.index_data)
ain = np.load(options.index_data)

head, tail = os.path.split(options.img_data)

if (bool(options.eval_for_original)):
    ain = ain.tolist()
    head, tail = os.path.split(options.img_data)
    feat = np.load(options.img_data)
    print ("Loaded")
    f = open("data/image_features_"+"total"+".b","wb")
    for index,key in enumerate(ain) :
        f.write(key.encode())
        for i in feat[index]:
            f.write(bytearray(struct.pack("f", i)))
    f.close()
    cmd  = "make "+"total"+".out"
    ret = os.system(cmd)
    res.write(str(ret))

    print (ret)
    res.close()
    
'''
autoencode(options.img_data,options.img_data[:-7]+str(options.encoding_dim)+".pickle",int(options.encoding_dim),4096,options.optimizer,loss='binary_crossentropy',my_epochs=10)
deep_features_pth = options.img_data[:-7]+str(options.encoding_dim)+".pickle"

f = open(deep_features_pth,"rb")
deep_feat = pickle.load(f)
f.close()
f = open("data/image_features_"+tail[:-4]+".b","wb")
for index,key in enumerate(ain) :
    f.write(key.encode())
    for i in deep_feat[index]:
        f.write(bytearray(struct.pack("f", i)))
f.close()
cmd  = "make "+tail[:-4]+".out"
ret = os.system(cmd)
res.write(str(ret))
'''