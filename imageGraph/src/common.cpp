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

#include "common.hpp"
#include <sys/time.h>
using namespace std;

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
  FILE* f = fopen(p, m);
  if (!f)
  {
    printf("Failed to open %s\n", p);
    exit(1);
  }
  return f;
}

double clock_(void)
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

// Sparse * dense inner product
double inner(map<int, double>* x, double* w)
{
  double res = 0;
  for (map<int, double>::iterator it = x->begin(); it != x->end(); it ++)
    res += it->second * w[it->first];
  return res;
}

double inner(double* x, double* y, int K)
{
  double res = 0;
  for (int k = 0; k < K; k ++)
    res += x[k] * y[k];
  return res;
}

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double square(double x)
{
  return x*x;
}

double safeLog(double p)
{
  double x = log(1 + exp(p));
  if (isnan(x) or isinf(x))
  {
    if (isnan(p) or isinf(p))
    {
      printf("Bad prediction\n");
      exit(1);
    }
    return p;
  }
  return x;
}

edge::edge(int productFrom,
           int productTo,
           int label,
           int reverseLabel) :
    productFrom(productFrom),
    productTo(productTo),
    label(label),
    reverseLabel(reverseLabel)
  {}

edge::~edge()
{
}
