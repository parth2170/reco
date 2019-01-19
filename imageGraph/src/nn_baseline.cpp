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

#include "nn_baseline.hpp"

#include "HLBFGS.h"

#include "sys/time.h"
#include "limits"
#include "mahalanobis.hpp"

#include <cmath>
using namespace std;

/// HLBFGS regrettably requires that a global variable be passed around
extern nn_baseline* gtc;

void nn_baseline::init()
{
  gtc = this;

  featureDim = nManifest + corp->imFeatureDim;

  // total number of parameters
  NW = 1 + featureDim;

  // Initialize parameters and latent variables
  // Zero all weights
  W = new double [NW];
  for (int i = 0; i < NW; i++)
    W[i] = 0.01;
  W[0] = 1.0;

  parametersToFlatVector(W, &c, &logistic_edge, INIT);

  bestValidModel = new double [NW];

  for (int w = 0; w < NW; w ++)
    bestValidModel[w] = W[w];
}

/// Derivative of the log probability
double nn_baseline::l_dl(double* grad)
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

    // Just the weighted nearest-neighbor baseline, except that the parameters don't get updated
    *(dC[tid]) += (e->label - frac);
  }

  for (int t = 0; t < NT; t ++)
    delete[] f_space[t];
  delete[] f_space;

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
