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

#include "lrMahalanobis.hpp"
using namespace std;

extern lrMahalanobis* gtc;

/// Calculates the truncated sigmoid function
// f_space is space allocated to store the feature vector
double lrMahalanobis::prediction_edge(edge* e, double* f_space)
{
  vector<int> sparsity = featureForPair_edge(e->productFrom, e->productTo, f_space);

  double p = 0;

  for (int k = 0; k < K; k ++)
  {
    double pk = 0;
    for (vector<int>::iterator i = sparsity.begin(); i != sparsity.end(); i ++)
      pk += U[*i][k] * f_space[*i];
    p += pk*pk;
  }

  return *c - p;
}

void lrMahalanobis::init()
{
  gtc = this;

  featureDim = nManifest + corp->imFeatureDim;

  M = new double* [featureDim];
  for (int i = 0; i < featureDim; i ++)
    M[i] = new double [featureDim];

  // total number of parameters
  NW = 1 + featureDim*K;

  // Initialize parameters and latent variables
  // Zero all weights
  W = new double [NW];
  for (int i = 0; i < NW; i++)
    W[i] = 0;

  parametersToFlatVector(W, &c, &U, INIT);

  // Some kind of randomized initialization is required for the sake of tie-breaking
  for (int i = 0; i < featureDim; i ++)
    for (int k = 0; k < K; k ++)
      U[i][k] = 1.0 - rand() * 2.0 / RAND_MAX;

  bestValidModel = new double [NW];

  for (int w = 0; w < NW; w ++)
    bestValidModel[w] = W[w];
}

void lrMahalanobis::saveModel(char* savePath)
{
  FILE* f = fopen_(savePath, "w");
  fprintf(f, "{\n");
  fprintf(f, "  \"NW\": %d,\n", NW);
  fprintf(f, "  \"c\": %f,\n", *c);
  fprintf(f, "  \"U\":");
  fprintf(f, "  [\n");
  for (int i = 0; i < featureDim; i ++)
  {
    fprintf(f, "    [");
    for (int k = 0; k < K; k ++)
    {
      fprintf(f, "%f", U[i][k]);
      if (k < K - 1)
        fprintf(f, ", ");
    }
    fprintf(f, "]");
    if (i < featureDim - 1)
      fprintf(f, ",");
    fprintf(f, "\n");
  }
  fprintf(f, "  ],\n");
  fprintf(f, "  \"W\": [");
  for (int w = 0; w < NW; w ++)
  {
    fprintf(f, "%f", W[w]);
    if (w < NW - 1)
      fprintf(f, ", ");
  }
  fprintf(f, "]\n");
  fprintf(f, "}\n");
  fclose(f);
}


void lrMahalanobis::parametersToFlatVector(double* g,
                                           double** c,
                                           double*** U,
                                           action_t action)
{
  if (action == FREE)
  {
    delete[] *U;
    return;
  }

  if (action == INIT)
  {
    *U = new double* [featureDim];
  }

  int ind = 0;

  *c = g + ind;
  ind ++;

  for (int f = 0; f < featureDim; f ++)
  {
    (*U)[f] = g + ind;
    ind += K;
  }

  if (ind != NW)
  {
    printf("Got incorrect index (%d != %d) at line %d of mahalanobis.cpp\n", ind, NW, __LINE__);
    exit(1);
  }
}

/// Derivative of the log probability
double lrMahalanobis::l_dl(double* grad)
{
  double l_dlStart = clock_();

  #pragma omp parallel for
  for (int w = 0; w < NW; w ++)
    grad[w] = 0;

  int NT = omp_get_max_threads();

  // Separate gradient vectors for each thread
  double** gradT = new double* [NT];
  double** dC = new double* [NT];
  double*** dU = new double** [NT];

  double* llThread = new double [NT];
  for (int t = 0; t < NT; t ++)
    llThread[t] = 0;

  for (int t = 0; t < NT; t ++)
  {
    gradT[t] = new double [NW];
    for (int w = 0; w < NW; w ++)
      gradT[t][w] = 0;
    parametersToFlatVector(gradT[t], dC + t, dU + t, INIT);
  }

  double** f_space = new double* [NT];
  double** k_space = new double* [NT];
  for (int t = 0; t < NT; t ++)
  {
    f_space[t] = new double [featureDim];
    k_space[t] = new double [K];
  }

#pragma omp parallel for schedule(dynamic)
  for (int x = 0; x < validStart; x ++)
  {
    int tid = omp_get_thread_num();
    edge* e = edges[x];

    vector<int> sparsity = featureForPair_edge(e->productFrom, e->productTo, f_space[tid]);

    for (int k = 0; k < K; k ++)
      k_space[tid][k] = 0;

    for (vector<int>::iterator i = sparsity.begin(); i != sparsity.end(); i ++)
      for (int k = 0; k < K; k ++)
        k_space[tid][k] += f_space[tid][*i] * U[*i][k];

    double inp = 0;
    for (int k = 0; k < K; k ++)
      inp += k_space[tid][k]*k_space[tid][k];

    inp = *c - inp;

    if (e->label) llThread[tid] += inp;
    llThread[tid] -= safeLog(inp);

    double einp = exp(inp);
    double frac = einp / (1 + einp);

    for (vector<int>::iterator i = sparsity.begin(); i != sparsity.end(); i ++)
    {
      double fi = 2*f_space[tid][*i] * (e->label - frac);
      double* dUi = dU[tid][*i];
      for (int k = 0; k < K; k ++)
        dUi[k] -= fi * k_space[tid][k];
    }
    *(dC[tid]) += (e->label - frac);

    if (not ((x + 1) % 100000))
    {
      // Print progress so that you can see that something is happening...
      fprintf(stderr, "-");
      fflush(stdout);
    }
  }

  for (int t = 0; t < NT; t ++)
  {
    delete[] f_space[t];
    delete[] k_space[t];
  }
  delete[] f_space;
  delete[] k_space;

  for (int w = 1; w < NW; w ++)
    llThread[0] -= lambda*W[w]*W[w];

  double llTotal = 0;
  for (int t = 0; t < NT; t ++)
    llTotal += llThread[t];
  delete [] llThread;

  // Add up the gradients from all threads
  for (int t = 0; t < NT; t ++)
  {
    for (int w = 0; w < NW; w ++)
      grad[w] += gradT[t][w];
  }

  for (int w = 1; w < NW; w ++)
    grad[w] -= 2*lambda*W[w];

  for (int t = 0; t < NT; t ++)
  {
    delete [] gradT[t];
    parametersToFlatVector(0, dC + t, dU + t, FREE);
  }
  delete [] gradT;
  delete [] dC;
  delete [] dU;

  // fprintf(stderr, "took %f\n", clock_() - l_dlStart);

  return llTotal;
}
