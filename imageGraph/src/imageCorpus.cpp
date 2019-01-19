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

#include "HLBFGS.h"

#include "sys/time.h"
#include "limits"
#include "mahalanobis.hpp"

#include <cmath>
using namespace std;

/// HLBFGS regrettably requires that a global variable be passed around
imageCorpus* gtc;

/// Required function by HLBFGS library: computes the log-probability and gradient
void evalfunc(int N, double* x, double* prev_x, double* f, double* g)
{
  // Negative signs because we want gradient ascent rather than descent
  *f = -gtc->l_dl(g);
  for (int w = 0; w < N; w ++)
    g[w] *= -1;
}

/// Required function by HLBFGS library: prints progress
void newiteration(int iter, int call_iter, double* x, double* f, double* g, double* gnorm)
{
  fprintf(stderr, "X");
  fflush(stderr);
}

void imageCorpus::saveModel(char* savePath)
{
  FILE* f = fopen_(savePath, "w");
  fprintf(f, "{\n");
  fprintf(f, "  \"NW\": %d,\n", NW);
  fprintf(f, "  \"c\": %f,\n", *c);
  fprintf(f, "  \"beta\": [");
  for (int i = 0; i < featureDim; i ++)
  {
    fprintf(f, "%f", logistic_edge[i]);
    if (i < featureDim - 1)
      fprintf(f, ", ");
  }
  fprintf(f, "],\n");
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

void imageCorpus::init()
{
  gtc = this;

  featureDim = nManifest + corp->imFeatureDim;

  // total number of parameters
  NW = 1 + featureDim;

  // Initialize parameters and latent variables
  // Zero all weights
  W = new double [NW];
  for (int i = 0; i < NW; i++)
    W[i] = 0;

  parametersToFlatVector(W, &c, &logistic_edge, INIT);

  bestValidModel = new double [NW];

  for (int w = 0; w < NW; w ++)
    bestValidModel[w] = W[w];
}

/// Recover all parameters from a vector (g)
void imageCorpus::parametersToFlatVector(double* g,
                                         double** c,
                                         double** logistic_edge,
                                         action_t action) // Do the vectors need to be initialized
{
  if (action == FREE)
    return;

  int ind = 0;

  *c = g + ind;
  ind ++;

  *logistic_edge = g + ind;
  ind += featureDim;

  if (ind != NW)
  {
    printf("Got incorrect index (%d != %d) at line %d of imageCorpus.cpp\n", ind, NW, __LINE__);
    exit(1);
  }
}

/// Predict whether a potential edge (e) exists given the parameter values w
double imageCorpus::prediction_edge(edge* e, double* f_space)
{
  featureForPair_edge(e->productFrom, e->productTo, f_space);
  return *c - inner(f_space, logistic_edge, featureDim);
}

/// Derivative of the log probability
double imageCorpus::l_dl(double* grad)
{
  #pragma omp parallel for
  for (int w = 0; w < NW; w ++)
    grad[w] = 0;

  int NT = omp_get_max_threads();

  // Separate gradient vectors for each thread
  double** gradT = new double* [NT];
  double** dC = new double* [NT];
  double** dlogistic_edge = new double* [NT];

  double* llThread = new double [NT];
  for (int t = 0; t < NT; t ++)
    llThread[t] = 0;

  for (int t = 0; t < NT; t ++)
  {
    gradT[t] = new double [NW];
    for (int w = 0; w < NW; w ++)
      gradT[t][w] = 0;
    parametersToFlatVector(gradT[t], dC + t, dlogistic_edge + t, INIT);
  }

  double** f_space = new double* [NT];
  for (int t = 0; t < NT; t ++)
    f_space[t] = new double [featureDim];

#pragma omp parallel for
  for (int x = 0; x < validStart; x ++)
  {
    int tid = omp_get_thread_num();
    edge* e = edges[x];

    double inp = prediction_edge(e, f_space[tid]);

    if (e->label) llThread[tid] += inp;
    llThread[tid] -= safeLog(inp);

    double einp = exp(inp);
    double frac = einp / (1 + einp);

    for (int f = 0; f < featureDim; f ++)
      dlogistic_edge[tid][f] -= f_space[tid][f] * (e->label - frac);
    *(dC[tid]) += (e->label - frac);
  }

  for (int t = 0; t < NT; t ++)
    delete[] f_space[t];
  delete[] f_space;

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
    parametersToFlatVector(0, dC + t, dlogistic_edge + t, FREE);
  }
  delete [] gradT;
  delete [] dC;
  delete [] dlogistic_edge;

  return llTotal;
}

/// Compute the training, validation, and test error
double imageCorpus::error(error_t err)
{
  int NT = omp_get_max_threads();

  double** f_space = new double* [NT];
  for (int t = 0; t < NT; t ++) f_space[t] = new double [featureDim];

  double* res_thread = new double [NT];
  for (int t = 0; t < NT; t ++)
    res_thread[t] = 0;

  int start = err == TRAIN ? 0 : err == VALID ? validStart : testStart;
  int end = err == TRAIN ? validStart : err == VALID ? testStart : nEdges;

#pragma omp parallel for
  for (int i = start; i < end; i ++)
  {
    int tid = omp_get_thread_num();
    edge* e = edges[i];

    double p1 = prediction_edge(e, f_space[tid]);
    if ((p1 > 0 and not e->label) or // Predicted an edge when there wasn't one
       (p1 <= 0 and e->label)) // Didn't predict an edge when there was one
      res_thread[tid] ++;
  }

  double res = 0;
  for (int t = 0; t < NT; t ++)
  {
    res += res_thread[t];
    delete[] f_space[t];
  }
  delete[] f_space;

  res /= end - start;

  delete [] res_thread;
  return res;
}

/// Train a model for "emIterations" with "gradIterations" of gradient descent at each step
void imageCorpus::train(int gradIterations)
{
  //TODO: fix this so that it stops after the iteration that minimizes the validation error
  double parameter[20];
  int info[20];
  //initialize
  INIT_HLBFGS(parameter, info);
  info[4] = gradIterations;
  info[5] = 0;
  info[6] = 0;
  info[7] = 0;
  info[10] = 0;
  info[11] = 1;
  HLBFGS(NW, 20, W, evalfunc, 0, HLBFGS_UPDATE_Hessian, newiteration, parameter, info);
  fprintf(stderr, "\n");
}
