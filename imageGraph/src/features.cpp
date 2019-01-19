/*
 * This software is Copyright © 2016 The Regents of the University of California. All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission to make commercial use of this software may be obtained by contacting:
 *
 * Office of Innovation & Commercialization
 * 9500 Gilman Drive, Mail Code 0910
 * University of California
 * La Jolla, CA 92093-0910
 * (858) 534-5815
 * invent@ucsd.edu
 * This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.
 *
 *IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#include "imageCorpus.hpp"
using namespace std;

vector<int> imageCorpus::featureForPair_edge(int productFrom, int productTo, double* feature)
{
  int ind = 0;
  vector<int> sparsity;
  // feature[ind++] = 1; // Offset is handled elsewhere
 
  if (nManifest != ind)
  {
    printf("Got unexpected number of manifest features on line %d (%d != %d)\n", __LINE__, nManifest, ind);
    exit(1);
  }

  for (int f = 0; f < corp->imFeatureDim; f ++)
  {
    feature[ind] = corp->imageFeatures[productTo][f] - corp->imageFeatures[productFrom][f];
    if (feature[ind] != 0) sparsity.push_back(ind);
    ind ++;
  }
  return sparsity;
}

void imageCorpus::initEdges()
{
  srand(0);

  edges = vector<edge*>();

  set<pair<int,int> >* G = &(corp->productGraphs[which_graph]);

  for (set<pair<int,int> >::iterator it = G->begin(); it != G->end(); it ++)
  {
    int label = 1;
    int productFrom = it->first;
    int productTo = it->second;
    int reverseLabel = 0;

    if (G->find(make_pair(productTo,productFrom)) != G->end())
      reverseLabel = 1;
    edge* e = new edge(productFrom,
                       productTo,
                       label,
                       reverseLabel);
    edges.push_back(e);
  }

  int NN = corp->nodesInSomeEdgeV.size();

  while (edges.size() < 2*G->size())
  {
    int label = 0;
    int reverseLabel = 0;
    int productFrom = corp->nodesInSomeEdgeV[rand() % NN];
    int productTo = corp->nodesInSomeEdgeV[rand() % NN];
    if (productFrom == productTo or G->find(make_pair(productFrom,productTo)) != G->end())
      continue;
    if (G->find(make_pair(productTo,productFrom)) != G->end())
      reverseLabel = 1;
    edge* e = new edge(productFrom,
                       productTo,
                       label,
                       reverseLabel);
    edges.push_back(e);
  }

  random_shuffle(edges.begin(), edges.end());
  nEdges = edges.size();
}
