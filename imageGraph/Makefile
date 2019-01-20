CC = g++
CFLAGS = -Wall -O3 -fopenmp -Igzstream -Isrc -IHLBFGS -Icpp/src
LDFLAGS = -lgomp -lgzstream -lz -lstdc++ -Lgzstream
OBJECTS = obj/common.o obj/categoryTree.o obj/corpus.o obj/features.o obj/HLBFGS.o obj/HLBFGS_BLAS.o obj/LineSearch.o obj/ICFS.o gzstream/gzstream.o
MODELOBJECTS = obj/imageCorpus.o obj/mahalanobis.o obj/lrMahalanobis.o obj/nn_baseline.o

CATEGORYPATH = categoryFilesNoReviews
#regularization constant
BANDWIDTH = 1

#used for topic models, can be ignored for now
PRODUCTSPERTOPIC = 1000

all: train outfits

gzstream/gzstream.o:
	cd gzstream && make

obj/common.o: src/common.hpp src/common.cpp Makefile
	$(CC) $(CFLAGS) -c src/common.cpp -o $@

obj/categoryTree.o: src/categoryTree.hpp src/categoryTree.cpp gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -c src/categoryTree.cpp -o $@

obj/corpus.o: src/corpus.hpp src/corpus.cpp obj/common.o obj/categoryTree.o gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -c src/corpus.cpp -o $@

obj/HLBFGS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/HLBFGS.cpp -o $@

obj/HLBFGS_BLAS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/HLBFGS_BLAS.cpp -o $@

obj/LineSearch.o:
	$(CC) -O3 -fopenmp -c HLBFGS/LineSearch.cpp -o $@

obj/ICFS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/ICFS.cpp -o $@

obj/imageCorpus.o: src/imageCorpus.hpp src/imageCorpus.cpp obj/features.o obj/corpus.o obj/common.o obj/HLBFGS.o Makefile
	$(CC) $(CFLAGS) -c src/imageCorpus.cpp -o $@

obj/nn_baseline.o: src/nn_baseline.hpp src/nn_baseline.cpp obj/imageCorpus.o obj/features.o obj/corpus.o obj/common.o obj/HLBFGS.o Makefile
	$(CC) $(CFLAGS) -c src/nn_baseline.cpp -o $@

obj/mahalanobis.o: src/mahalanobis.hpp src/mahalanobis.cpp obj/imageCorpus.o obj/features.o obj/corpus.o obj/common.o obj/HLBFGS.o Makefile
	$(CC) $(CFLAGS) -c src/mahalanobis.cpp -o $@

obj/lrMahalanobis.o: src/lrMahalanobis.hpp src/lrMahalanobis.cpp obj/mahalanobis.o obj/features.o obj/corpus.o obj/common.o obj/HLBFGS.o Makefile
	$(CC) $(CFLAGS) -c src/lrMahalanobis.cpp -o $@

obj/features.o: src/features.cpp src/imageCorpus.hpp obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/features.cpp -o $@

train: src/main.cpp $(OBJECTS) $(MODELOBJECTS) gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -o train src/main.cpp $(OBJECTS) $(MODELOBJECTS) $(LDFLAGS)

outfits: src/outfits.cpp $(OBJECTS) $(MODELOBJECTS) gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -o outfits src/outfits.cpp $(OBJECTS) $(MODELOBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(MODELOBJECTS) gzstream/gzstream.o train outfits

ADIR = data/

%.out: train
	./train \
	0 \
	$(ADIR)/productMeta.txt.gz \
	$(ADIR)/duplicate_list.txt.gz \
	$(ADIR)/image_features_$*.b \
	5 \
	$(ADIR)/also_viewed.txt.gz
