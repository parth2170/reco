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

#include "corpus.hpp"
#include "nn_baseline.hpp"
#include "imageCorpus.hpp"
#include "mahalanobis.hpp"
#include "lrMahalanobis.hpp"
using namespace std;

/// Passes around the parameters to run one experiment
void experiment(corpus* corp, // Data
                double lambda, // Regularization parameter (always 0 in the SIGIR paper)
                int iterations, // Gradient iterations
                char* modelPath, // Where to save the model
                int g, // Which graph to use
                int K) // Model dimensionality
{
  printf("  \"K\": %d,\n", K);
  printf("  \"lambda\": %f,\n", lambda);

  double train = 0; 
  double valid = 0;
  double test = 0;

  if (K > 0)
  {
    lrMahalanobis ec(corp, g, lambda, 0, K);
    ec.init();
    ec.train(iterations);

    train = ec.error(TRAIN);
    valid = ec.error(VALID);
    test = ec.error(TEST);

    char* mp = new char [10000];
    // Use this code if you want to save the model somewhere
    if (modelPath)
    {
      sprintf(mp, "%s-%s-%d-%f.txt", modelPath, corp->graphNames[g].c_str(), K, lambda);
      ec.saveModel(mp);
    }
    delete [] mp;
  }
  else // 0-dimensional model is treated as the nearest-neighbor baseline
  {
    nn_baseline nn(corp, g, lambda, 0);
    nn.init();
    nn.train(iterations);

    train = nn.error(TRAIN);
    valid = nn.error(VALID);
    test = nn.error(TEST);
  }

  printf("  \"error\": {\"train\": %f, \"valid\": %f, \"test\": %f}\n", train, valid, test);
}

int main(int argc, char** argv)
{
  srand(0);

  if (argc < 7)
  {
    printf("Files required are:\n");
    printf("  1: l1 regularization parameter (0 in SIGIR)\n");
    printf("  2: metadata file\n");
    printf("  3: list of potential duplicate products to be merged\n");
    printf("  4: image features\n");
    printf("  5: transform dimensionality\n");
    printf("  6: graph\n");
    exit(0);
  }
  double lambda = atof(argv[1]);
  char* categoryPath = argv[2];
  char* duplicatePath = argv[3];
  char* imFeaturePath = argv[4];
  int K = atoi(argv[5]);

  // char* predictionPath = 0;

  printf("{\n");
  printf("  \"corpus\": \"%s\",\n", argv[5]);

  int G = argc - 6;

  corpus corp("null", // review path
              argv + 6, // graph paths
              categoryPath, // path to category data
              duplicatePath, // path to duplicate ASIN listing
              imFeaturePath, // path to image features
              G, // number of graphs
              1000,
              0); // max number of reviews to read (0 for all)

  experiment(&corp, lambda, 500, "Y.txt", 0, K);

  printf("}\n");
  return 0;
}
