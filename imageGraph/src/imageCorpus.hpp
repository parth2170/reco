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

#pragma once
#include "common.hpp"
#include "corpus.hpp"

void evalfunc(int N, double* x, double* prev_x, double* f, double* g);
void newiteration(int iter, int call_iter, double* x, double* f, double* g, double* gnorm);

class imageCorpus
{
public:
  imageCorpus(corpus* corp, // The corpus
              int which_graph, // The number of graphs
              double lambda, // regularization parameter
              int nManifest // Number of manifest edge features
             ) :
    corp(corp),
    which_graph(which_graph),
    lambda(lambda),
    nManifest(nManifest)
  {
    srand(0);
    nItems = corp->nItems;
    initEdges();

    double testFraction = 0.1;
    validStart = (int) ((1.0 - 2*testFraction)*nEdges);
    testStart = (int) ((1.0 - testFraction)*nEdges);

    if (validStart > 2000000)
    {
      validStart = 2000000;
      int stillHave = nEdges - validStart;
      testStart = validStart + stillHave / 2;
    }

    if (validStart < 1 or (testStart - validStart) < 1 or (nEdges - testStart) < 1)
    {
      printf("Didn't get enough edges (%d/%d/%d)\n", validStart, testStart, nEdges);
      exit(1);
    }
  }

  ~imageCorpus()
  {
    parametersToFlatVector(0, &c, &logistic_edge, FREE);
    delete[] W;

    for (std::vector<edge*>::iterator it = edges.begin(); it != edges.end(); it ++)
      delete *it;
    delete [] bestValidModel;
  }

  virtual void init();
  virtual void saveModel(char* savePath);

  virtual double prediction_edge(edge* e, double* f_space);
  std::vector<int> featureForPair_edge(int productFrom, int productTo, double* feature);

  void initEdges(void);

  virtual void parametersToFlatVector(double* g,
                                      double** c,
                                      double** logistic_edge,
                                      action_t action);

  virtual double l_dl(double* grad);
  void train(int gradIterations);
  double error(error_t err);

  void savePrecisionRecall(char* PRPath, int limit);
  void saveTopK(char* topKPath, int sampleSize);

  corpus* corp;
  int which_graph;
  double* bestValidModel;

  int validStart;
  int testStart;

  int featureDim;

  // Model parameters
  double* logistic_edge; // Logistic parameters (F)
  double* c; // Scale factor on distance transform
  double* W; // Contiguous version of all parameters, i.e., a flat vector containing all parameters in order (useful for lbfgs)
  int NW;

  double lambda;

  std::vector<edge*> edges;
  int nManifest;

  int nItems; // Number of items
  int nEdges; // Number of edges
};
