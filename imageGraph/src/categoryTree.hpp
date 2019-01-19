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

#include "map"
#include "vector"
#include "string"
#include "set"

/// A node of the category tree
class categoryNode
{
public:
  categoryNode(std::string name, categoryNode* parent, int nodeId, int productsPerTopic);
  ~categoryNode();

  void observe(void);
  int howManyTopics(void);
  void addChild(categoryNode* child);

  categoryNode* find(std::string* category, int L);

  void print(int depth);
  void fprintJson(int depth, FILE* f);

  std::string name;
  categoryNode* parent;
  std::map<std::string, categoryNode*> children;
  int productCount; // How many products belong to this category?
  std::set<int> productSet;
  int nodeId;
  int productsPerTopic;
};

/// A complete category hierarchy
class categoryTree
{
public:
  categoryTree(std::string rootName, bool skipRoot, int productsPerTopic);
  categoryTree();
  ~categoryTree();

  void print(void);
  void fprintJson(FILE* f);

  categoryNode* addPath(std::vector<std::string> category);

  std::vector<categoryNode*> pathFromId(int nodeId);
  void incrementCounts(int nodeId);
  int nNodes(void);

  bool skipRoot; // skipRoot should be true if there are multiple "top-level" categories, i.e., if the root node is not a "real" category.
  categoryNode* root;
  int productsPerTopic;

  std::map<categoryNode*,int> nodeToId;
  std::vector<categoryNode*> idToNode;
};

/// Mapping between nodes in a category tree and topics
class topicTree
{
public:
  topicTree(categoryTree* ct);

  int totalTopics;
  std::vector<int> topicStart; // Maps a node (category) to its first topic
  std::vector<int> topicCount; // Maps a node to the number of contiguous topics it contains
  std::vector<int> topicNode; // Maps a topic to its node
  std::map<int,int> topicParent; // Maps a topic to its parent topic

  categoryTree* ct;
};
