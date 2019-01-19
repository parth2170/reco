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
#include "categoryTree.hpp"

class corpus
{
public:
  corpus(std::string voteFile, char** graphPaths, char* categoryPath, char* duplicatePath, char* featurePath, int G, int productsPerTopic, int max);
  ~corpus();
  
  void loadGraphs(char** graphPaths, char* duplicatePath);
  void loadCategories(char* categoryPath, std::string rootName, bool skipRoot);

  std::vector<vote*>* V;

  int nUsers; // Number of users
  int nItems; // Number of items
  int nWords; // Number of words
  // int nEdges; // Number of edges
  int G; // Number of graphs
  int productsPerTopic;

  std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> itemIds; // Maps an item's string-valued ID to an integer

  std::map<int, std::string> rUserIds; // Inverse of the above maps
  std::map<int, std::string> rItemIds;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> wordId; // Map each word to its integer ID
  std::map<int, std::string> idWord; // Inverse of the above map

  std::vector<std::string> graphNames; // Names of each graph type
  std::set<std::pair<int,int> >* productGraphs; // Edgelist per graph
  std::map<int, std::vector<int> >* edgeAdjGraphs; // Edge-adjacency list per graph

  std::set<int> nodesInSomeEdge; // Set of nodes that appear in some edge
  std::vector<int> nodesInSomeEdgeV; // Same thing as a vector

  std::set<std::string> adjectives;
  std::set<std::string> nouns;
  std::set<std::string> brands;

  categoryTree* ct; // Category hierarchy
  std::map<int, std::set<int> > productCategories; // Set of categories of each product (products can belong to multiple categories)
  std::map<int, std::string> itemBrand;
  std::map<int, double> itemPrice;

  std::vector<float*> imageFeatures;
  int imFeatureDim; // Dimensionality of the image features (hardcoded!)
};
