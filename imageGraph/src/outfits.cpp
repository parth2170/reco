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
#include "imageCorpus.hpp"
#include "mahalanobis.hpp"
#include "lrMahalanobis.hpp"
#include <stdio.h>
using namespace std;

/// Check if a string starts with a character
bool startsWith(const char* s, string sub)
{
  int i = 0;
  while (sub.c_str()[i] != '\0' and s[i] != '\0')
  {
    if (s[i] != sub.c_str()[i]) return false;
    i ++;
  }
  if (sub[i] == '\0') return true;
  return false;
}

/// Load a pre-trained model from a json file
double* loadModel(char* modelPath, int* NW)
{
  ifstream in;
  in.open(modelPath);

  string line;

  double* W = 0;
  int i = 0;

  while (getline(in, line))
  {
    const char* lb = line.c_str();
    if (startsWith(lb, "  \"NW\""))
    {
      sscanf(lb, "  \"NW\": %d", NW);
      W = new double [*NW];
    }

    if (startsWith(lb, "  \"W\""))
    {
      int consumed;
      sscanf(lb, "  \"W\": [%n", &consumed);
      lb += consumed;
      float f;
      while (sscanf(lb, "%f%n", &f, &consumed) == 1)
      {
        W[i++] = f;
        lb += consumed;
        if (sscanf(lb, ",%n", &consumed) != 0)
          break;
        lb += consumed;
      }
    }
  }
  if (i != *NW)
  {
    printf("Read unexpected number of parameters (%d != %d)\n", i, *NW);
    exit(1);
  }

  in.close();
  return W;
}

/// Squared Euclidean distance between features, used after the high-dimensional features have already been projected
double dist(double* x1, double* x2, int K)
{
  double d = 0;
  for (int k = 0; k < K; k ++)
    d += square(x2[k] - x1[k]);
  return d;
}

// Find a single item from a particular category that goes well with the query
int query(lrMahalanobis* ec, // Trained model object
          double** Uc, // Features for each item projected into K dimensions
          int qItem, // The id of the query item
          categoryNode* queryNode, // The category of the item we want returned
          map<categoryNode*, vector<int> >* categoryVecs) // Maps categories to sets of items belonging to that category
{
  int NT = omp_get_max_threads();

  double* bestDt = new double [NT];
  int* bestItemt = new int [NT];

  for (int t = 0; t < NT; t ++)
  {
    bestDt[t] = 0;
    bestItemt[t] = -1;
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) categoryVecs->at(queryNode).size(); i ++)
  {
    int item = categoryVecs->at(queryNode)[i];
    double d = dist(Uc[qItem], Uc[item], ec->K); // Find the distance between the query item and the candidate item i
    int tid = omp_get_thread_num();
    if (d > 0 and (bestItemt[tid] == -1 or d < bestDt[tid]))
    {
      bestDt[tid] = d;
      bestItemt[tid] = item;
    }
  }

  double bestD = 0;
  int bestItem = -1;

  for (int t = 0; t < NT; t ++)
  {
    if (bestItemt[t] == -1)
      // This thread didn't do anything
      continue;
    if (bestItem == -1 or bestDt[t] < bestD)
    {
      bestD = bestDt[t];
      bestItem = bestItemt[t];
    }
  }

  if (bestItem == -1)
  {
    fprintf(stderr, "NO ITEM: %d %d\n", qItem, (int) categoryVecs->at(queryNode).size());
  }

  return bestItem;
}

/// Generate an entire outfit by running the above function for each of a set of categories
vector<int> queriesProductCategory(lrMahalanobis* ec, double** Uc, int item, vector<categoryNode*> categories, map<categoryNode*, vector<int> >* categoryVecs)
{
  vector<int> res;
  for (vector<categoryNode*>::iterator it = categories.begin(); it != categories.end(); it ++)
  {
    if (ec->corp->productCategories.find(item) == ec->corp->productCategories.end())
    {
      // Item doesn't have a category
      printf("This shouldn't happen (line %d)\n", __LINE__);
      exit(1);
    }
    // One query per category of item we want returned
    res.push_back(query(ec, Uc, item, *it, categoryVecs));
  }
  return res;
}

/// Project the features into low-dimensions in advance
double** projectFeatures(lrMahalanobis* ec, double* W)
{
  ec->parametersToFlatVector(W, &(ec->c), &(ec->U), COPY);

  int nItems = ec->nItems;
  int K = ec->K;

  double** Uc = new double* [nItems];
  Uc[0] = new double [nItems * K];
  for (int i = 0; i < nItems; i ++)
  {
    Uc[i] = *Uc + i*K;
    for (int k = 0; k < K; k ++)
      Uc[i][k] = 0;

    for (int f = 0; f < ec->corp->imFeatureDim; f ++)
      for (int k = 0; k < ec->K; k ++)
        Uc[i][k] += ec->U[f][k] * ec->corp->imageFeatures[i][f];
  }

  return Uc;
}

/// Build a html file with a bunch of outfits in it
void queriesFromCategories(lrMahalanobis* ec, // Model object
                           double* W, // Flattened parameter vector
                           vector<vector<string> > categoriesS, // Set of categories that defines an "outfit"
                           map<string, string>* urls, // Maps ASINs to image URLs
                           int NQ) // Number of outfits to generate
{
  double** Uc = projectFeatures(ec, W);
  vector<categoryNode*> categories;
  set<int> allProductsInCategories;
  map<categoryNode*, vector<int> > categoryVecs;
  
  for (vector<vector<string> >::iterator it = categoriesS.begin(); it != categoriesS.end(); it ++)
  {
    categoryNode* cn = ec->corp->ct->addPath(*it);
    if (cn->productSet.size() == 0)
    {
      printf("No products for category.\n");
      for (vector<string>::iterator it2 = it->begin(); it2 != it->end(); it2 ++)
        printf("  %s\n", it2->c_str());
      exit(1);
    }
    categories.push_back(cn);
    categoryVecs[cn] = vector<int>();
    for (set<int>::iterator it2 = cn->productSet.begin(); it2 != cn->productSet.end(); it2 ++)
    {
      string asin = ec->corp->rItemIds[*it2];
      if (urls->find(asin) == urls->end()) continue;
      allProductsInCategories.insert(*it2);
      categoryVecs[cn].push_back(*it2);
    }
  }

  vector<int> allProductsInCategoriesV;
  for (set<int>::iterator it = allProductsInCategories.begin(); it != allProductsInCategories.end(); it ++)
    allProductsInCategoriesV.push_back(*it);

  printf("-->\n");
  printf("<!DOCTYPE html><html><body>\n");
  printf("<table>\n");

  for (int q = 0; q < NQ; q ++)
  {
    categoryNode* cn = categories[rand() % categories.size()];

    int item = categoryVecs[cn][rand() % categoryVecs[cn].size()];
    string asin = ec->corp->rItemIds[item];
    if (urls->find(asin) == urls->end())
    {
      q --;
      continue;
    }
    string url = urls->at(asin);
    printf("<tr><td><h1>Query:</h1></td><td><h1>Outfit:</h1></td></tr>\n");
    printf("<tr><td><a href=\"http://amazon.com/dp/%s/\"><img src=\"%s\"></a></td>", asin.c_str(), url.c_str());
    printf("<td>");
    vector<int> outfit = queriesProductCategory(ec, Uc, item, categories, &categoryVecs);
    for (vector<int>::iterator it = outfit.begin(); it != outfit.end(); it ++)
    {
      string asin = ec->corp->rItemIds[*it];
      if (urls->find(asin) == urls->end())
      {
        fprintf(stderr, "Still missing an asin somehow...\n");
        continue;
      }
      string url = urls->at(asin);
      printf("<a href=\"http://amazon.com/dp/%s/\"><img src=\"%s\"></a>", asin.c_str(), url.c_str());
    }
    printf("</td></tr>\n");
  }

  printf("</table>\n");
  printf("</body></html>\n");

  delete [] *Uc;
  delete [] Uc;
}

/// Outfits for men's clothing
void queriesMen(lrMahalanobis* ec, double* W, map<string, string>* urls, int NQ)
{
  vector<vector<string> > categoriesS;

  // Outfit = accessory + pants + shirt + shoes
  vector<string> q1;
  q1.push_back("Clothing Shoes & Jewelry");
  q1.push_back("Men");
  q1.push_back("Accessories");

  vector<string> q2;
  q2.push_back("Clothing Shoes & Jewelry");
  q2.push_back("Men");
  q2.push_back("Clothing");
  q2.push_back("Pants");

  vector<string> q3;
  q3.push_back("Clothing Shoes & Jewelry");
  q3.push_back("Men");
  q3.push_back("Clothing");
  q3.push_back("Shirts");

  vector<string> q4;
  q4.push_back("Clothing Shoes & Jewelry");
  q4.push_back("Men");
  q4.push_back("Shoes");

  categoriesS.push_back(q1);
  categoriesS.push_back(q2);
  categoriesS.push_back(q3);
  categoriesS.push_back(q4);

  queriesFromCategories(ec, W, categoriesS, urls, NQ);
}

/// Outfits for women's clothing
void queriesWomen(lrMahalanobis* ec, double* W, map<string, string>* urls, int NQ)
{
  vector<vector<string> > categoriesS;

  // Outfit = accessory + pants + top + shoes + jewelery + dress
  // (I know that not all of these things are mutually compatible)
  vector<string> q1;
  q1.push_back("Clothing Shoes & Jewelry");
  q1.push_back("Women");
  q1.push_back("Accessories");

  vector<string> q2;
  q2.push_back("Clothing Shoes & Jewelry");
  q2.push_back("Women");
  q2.push_back("Clothing");
  q2.push_back("Pants");

  vector<string> q3;
  q3.push_back("Clothing Shoes & Jewelry");
  q3.push_back("Women");
  q3.push_back("Clothing");
  q3.push_back("Tops & Tees");

  vector<string> q4;
  q4.push_back("Clothing Shoes & Jewelry");
  q4.push_back("Women");
  q4.push_back("Shoes");

  vector<string> q5;
  q5.push_back("Clothing Shoes & Jewelry");
  q5.push_back("Women");
  q5.push_back("Jewelry");

  vector<string> q6;
  q6.push_back("Clothing Shoes & Jewelry");
  q6.push_back("Women");
  q6.push_back("Clothing");
  q6.push_back("Dresses");

  categoriesS.push_back(q1);
  categoriesS.push_back(q2);
  categoriesS.push_back(q3);
  categoriesS.push_back(q4);
  categoriesS.push_back(q5);
  categoriesS.push_back(q6);

  queriesFromCategories(ec, W, categoriesS, urls, NQ);
}

/// Read urls from a file mapping ASINs to URLs
map<string, string>* loadUrls(char* urlPath)
{
  map<string, string>* urls = new map<string, string>();
  ifstream in;
  fprintf(stderr, "Loading urls from %s\n", urlPath);
  in.open(urlPath);

  string line;
  while (getline(in, line))
  {
    string asin;
    string imUrl;
    stringstream ss(line);
    ss >> asin >> imUrl;
    urls->insert(make_pair(asin, imUrl));
  }

  return urls;
}

int main(int argc, char** argv)
{
  char* categoryPath = argv[1]; // Metadata file (contains product category info)
  char* duplicatePath = argv[2]; // Products likely to be duplicates to filter out
  char* imFeaturePath = argv[3]; // Image features in binary format
  char* modelPath = argv[4]; // Pre-trained model file to load
  char* urlPath = argv[5]; // Image urls for html file
  
  // Not relevant, just here because I'm reusing code from KDD'15 paper
  int G = argc - 6;
  int productsPerTopic = 100;

  printf("<!--\n");

  corpus corp("null", // review path
              argv + 6, // graph paths
              categoryPath, // path to category data
              duplicatePath, // path to duplicate ASIN listing
              imFeaturePath, // path to image features
              G, // number of graphs
              productsPerTopic,
              0); // max number of reviews to read (0 for all)

  int NW;
  double* W = loadModel(modelPath, &NW);
  int K = NW / 4096;
  lrMahalanobis ec(&corp, 0, 0, 0, K);
  ec.init();

  map<string, string>* urls = loadUrls(urlPath);

  // Generate 1000 outfits
  if (string(imFeaturePath).find("Men") != string::npos)
  {
    printf("Running queries for Men's products\n");
    queriesMen(&ec, W, urls, 1000);
  }
  else if (string(imFeaturePath).find("Women") != string::npos)
  {
    printf("Running queries for Women's products\n");
    queriesWomen(&ec, W, urls, 1000);
  }

  delete [] W;
  delete urls;
}
