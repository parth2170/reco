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

/**
 This file is concerned with reading in review/rating data. It's not really so relevant to the SIGIR paper--all that's needed here is the category metadata and image features. It was easier to build this on top of code from the KDD 2015 paper which made use of reviews though.
 */

#include "corpus.hpp"
#include "common.hpp"
#include "categoryTree.hpp"

//#include <boost/algorithm/string/trim.hpp>
#include <ctype.h>
using namespace std;

/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(vote* v1, vote* v2)
{
  return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

static inline std::string &ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
  return ltrim(rtrim(s));
}

corpus::corpus(string voteFile, char** graphPaths, char* categoryPath, char* duplicatePath, char* featurePath, int G, int productsPerTopic, int max) :
    G(G), productsPerTopic(productsPerTopic)
{
  V = 0;
  ct = 0;

  FILE* f = fopen_(featurePath, "rb");

  char* asin = new char [11];
  asin[10] = '\0';

  nItems = 0;
  nUsers = 0;
  nWords = 0;

  imFeatureDim = 4096;

  int a;
  fprintf(stderr, "Loading image features from %s", featurePath);

  double ma = 58.388599; // Largest feature observed

  while (!feof(f))
  {
    if ((a = fread(asin, sizeof(*asin), 10, f)) != 10)
    {
      // printf("Expected to read %d chars, got %d\n", 10, a);
      break;
    }
    for (int c = 0; c < 10; c ++)
      if (not isascii(asin[c]))
      {
        printf("Expected asin to be 10-digit ascii\n");
        exit(1);
      }

    if (not (nItems % 1000))
    {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    float* feat = new float [imFeatureDim];
    if ((a = fread(feat, sizeof(*feat), imFeatureDim, f)) != imFeatureDim)
    {
      printf("Expected to read %d floats, got %d\n", imFeatureDim, a);
      exit(1);
    }

    for (int f = 0; f < imFeatureDim; f ++)
      feat[f] /= ma;

    string sAsin(asin);

    itemIds[sAsin] = nItems;
    rItemIds[nItems] = sAsin;

    nItems ++;

    imageFeatures.push_back(feat);
  }
  fprintf(stderr, "\n");

  delete[] asin;
  fclose(f);

  printf("  \"nItems\": %d,\n", (int) imageFeatures.size());
  loadGraphs(graphPaths, duplicatePath);
  loadCategories(categoryPath, "root", false);
}

/// Parse G product graphs
void corpus::loadGraphs(char** graphPaths, char* duplicatePath)
{
  fprintf(stderr, "Loading duplicate labels from %s", duplicatePath);
  map<string, string> duplicates;
  igzstream inDup;
  inDup.open(duplicatePath);
  string asin;
  string dupof;

  int count = 0;
  while (inDup >> asin >> dupof)
  {
    duplicates[asin] = dupof;
    count ++;
    if (not (count % 100000)) fprintf(stderr, ".");
  }

  inDup.close();
  fprintf(stderr, "\n");

  productGraphs = new set<pair<int,int> > [G];
  edgeAdjGraphs = new map<int, vector<int> > [G];

  fprintf(stderr, "Loading graphs");

  printf("  \"nEdges\": {");

  for (int g = 0; g < G; g ++)
  {
    igzstream in;
    in.open(graphPaths[g]);

    string n1;
    string n2;
    string edgename;
    
    string line;

    while (getline(in, line))
    {
      stringstream ss(line);
      ss >> n1 >> edgename; // Second word of each line should be the edge type
      if (itemIds.find(n1) == itemIds.end())
        continue;
      int bid1 = itemIds[n1];
      if (edgeAdjGraphs[g].find(bid1) == edgeAdjGraphs[g].end())
        edgeAdjGraphs[g][bid1] = vector<int>();

      while (ss >> n2)
      {
        if (itemIds.find(n2) == itemIds.end())
        {
          if (duplicates.find(n2) != duplicates.end())
            n2 = duplicates[n2];
          if (itemIds.find(n2) == itemIds.end())
            continue;
        }
        int bid2 = itemIds[n2];

        productGraphs[g].insert(make_pair(bid1,bid2));
        edgeAdjGraphs[g][bid1].push_back(bid2);

        if (not (productGraphs[g].size() % 1000))
        {
          fprintf(stderr, ".");
          fflush(stdout);
        }

        nodesInSomeEdge.insert(bid1);
        nodesInSomeEdge.insert(bid2);
      }
    }

    fprintf(stderr, "\n");
    graphNames.push_back(edgename);
    printf("\"%s\": %d", edgename.c_str(), (int) productGraphs[g].size());
    if (g < G - 1) printf(", ");
  }
  printf("},\n");


  for (set<int>::iterator it = nodesInSomeEdge.begin(); it != nodesInSomeEdge.end(); it ++)
    nodesInSomeEdgeV.push_back(*it);
}

/// Parse category info for all products
void corpus::loadCategories(char* categoryPath, string rootName, bool skipRoot)
{
  fprintf(stderr, "Loading category data");
  ct = new categoryTree(rootName, skipRoot, productsPerTopic);

  igzstream in;
  in.open(categoryPath);

  string line;

  int currentProduct = -1;
  int count = 0;

  while (getline(in, line))
  {
    istringstream ss(line);

    if (line.c_str()[0] != ' ')
    {
      string itemId;
      double price = -1;
      string brand("unknown_brand");
      ss >> itemId;
      ss >> price >> brand;
      if (itemIds.find(itemId) == itemIds.end())
        currentProduct = -1; // Invalid product
      else
      {
        currentProduct = itemIds[itemId];
        productCategories[currentProduct] = set<int>();

        if (brand.compare("unknown_brand") != 0)
          itemBrand[currentProduct] = brand;
        if (price > 0)
          itemPrice[currentProduct] = price;
      }
      count ++;
      if (not (count % 100000)) fprintf(stderr, ".");
      continue;
    }

    vector<string> category;

    // Category for each product is a comma-separated list of strings
    string cat;
    while (getline(ss, cat, ','))
    {
      category.push_back(trim(cat));
    }

    if (currentProduct != -1)
    {
      categoryNode* child = ct->addPath(category);
      child->productSet.insert(currentProduct);

      //Don't do this: this would mean we'd double-count top-level categories from products that appear in multiple categories
      // ct->incrementCounts(child->nodeId);

      categoryNode* current = child;
      while (current)
      {
      	if (productCategories[currentProduct].find(current->nodeId) == productCategories[currentProduct].end())
      	{
          productCategories[currentProduct].insert(current->nodeId);
          current->observe();
        }
        current = current->parent;
      }
    }
  }

  fprintf(stderr, "\n");
  in.close();
  ct->print();
}

corpus::~corpus()
{
  if (V)
  {
    for (vector<vote*>::iterator it = V->begin(); it != V->end(); it++)
      delete *it;
    delete V;
  }

  delete [] productGraphs;
  delete [] edgeAdjGraphs;
  if (ct) delete ct;

  for (vector<float*>::iterator it = imageFeatures.begin(); it != imageFeatures.end(); it ++)
    delete [] *it;
}
