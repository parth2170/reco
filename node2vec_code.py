import sys
import pickle
import networkx as nx 
from node2vec import Node2Vec

network_path = sys.argv[1]
outfiles = sys.argv[2]

def run_save():
	graph = nx.read_gpickle(network_path+"network.gpickle")
	node2vec = Node2Vec(graph, dimensions=300, walk_length=30, num_walks=500, workers=4) 
	model = node2vec.fit(window=8, min_count=1, batch_words=5)
	model.wv.save_word2vec_format(outfiles+"sample_node2vec_embeddings")
	model.save(outfiles+"sample_model")

	with open(outfiles+'sample_paths.pickle', 'wb') as fp:
	    pickle.dump(node2vec.walks, fp, protocol=pickle.HIGHEST_PROTOCOL)

	with open(outfiles+'sample_walks.txt', 'w') as f:
	    for item in node2vec.walks:
	        f.write("%s\n" % item)

run_save()