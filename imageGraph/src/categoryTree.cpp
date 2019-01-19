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

/// This file is concerned with reading in category metadata (basically the category hierarchy, and each product's position in it)

#include "categoryTree.hpp"

#include "algorithm"
#include "map"
#include "set"
#include "vector"
#include "string"
#include "stdlib.h"
#include "stdio.h"
#include "sstream"
#include "fstream"

//#include <boost/algorithm/string/trim.hpp>
using namespace std;

categoryNode::categoryNode(string name, categoryNode* parent, int nodeId, int productsPerTopic) :
    name(name),
    parent(parent),
    productCount(0),
    nodeId(nodeId),
    productsPerTopic(productsPerTopic)
{
  children = map<string, categoryNode*>();
  productSet = set<int>();
}

categoryNode::~categoryNode()
{
  for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++)
    delete it->second;
}

/// Should be called every time a product from this category is observed when loading the corpus
void categoryNode::observe(void)
{
  productCount ++;
}

/// How many topics should be associated with a particular category (based on the number of times it was observed in the corpus)
int categoryNode::howManyTopics()
{
  if (nodeId == 0)
    return 10; // at least 10 for the root node, no matter what
  if (productCount < productsPerTopic)
    return 0;
  if (productCount > 10*productsPerTopic)
    return 10;
  return productCount / productsPerTopic;
}

void categoryNode::addChild(categoryNode* child)
{
  children[child->name] = child;
}

/// Walk down a category tree looking for a particular category (list of strings of length L), or return 0 if it doesn't exist
categoryNode* categoryNode::find(string* category, int L)
{
  if (L == 0) return this;
  if (children.find(category[0]) == children.end()) return 0;
  return children[category[0]]->find(category + 1, L - 1);
}

void categoryNode::print(int depth)
{
  if (howManyTopics() > 0)
  { // Ignore rare categories just for brevity
    for (int d = 0; d < depth; d ++)
      fprintf(stderr, "  ");
  fprintf(stderr, "%d (%s), count = %d\n", nodeId, name.c_str(), productCount);
  }
  for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++)
    (it->second)->print(depth + 1);
}

/// Print a JSON representation of the category tree to a file
void categoryNode::fprintJson(int depth, FILE* f)
{
  for (int d = 0; d < depth; d ++) fprintf(f, "  ");
  fprintf(f, "{");

  fprintf(f, "\"nodeId\": %d, \"nodeName\": \"%s\", \"observations\": %d", nodeId, name.c_str(), productCount);
  if (children.size() > 0)
  {
    bool childTopics = false;
    for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++)
      if (it->second->howManyTopics())
      {
        childTopics = true;
        break;
      }
    if (childTopics)
    {
    	fprintf(f, ", \"children\":\n");
      for (int d = 0; d < depth + 1; d ++) fprintf(f, "  ");
      fprintf(f, "[\n");
      bool first = true;
    	for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++)
    	{
        if (it->second->howManyTopics())
        {
      	  if (not first) fprintf(f, ",\n");
          it->second->fprintJson(depth + 2, f);
          first = false;
        }
    	}
      fprintf(f, "\n");
      for (int d = 0; d < depth + 1; d ++) fprintf(f, "  ");
      fprintf(f, "]");
    }
  }
  fprintf(f, "\n");
  for (int d = 0; d < depth; d ++) fprintf(f, "  ");
  fprintf(f, "}");
}

categoryTree::categoryTree(string rootName, bool skipRoot, int productsPerTopic) : skipRoot(skipRoot), productsPerTopic(productsPerTopic)
{
  root = new categoryNode(rootName, 0, 0, productsPerTopic);

  nodeToId = map<categoryNode*, int>();
  idToNode = vector<categoryNode*>();

  nodeToId[root] = 0;
  idToNode.push_back(root);
}

categoryTree::categoryTree() : skipRoot(false), productsPerTopic(1000)
{
  root = new categoryNode("root", 0, 0, productsPerTopic);

  nodeToId = map<categoryNode*, int>();
  idToNode = vector<categoryNode*>();

  nodeToId[root] = 0;
  idToNode.push_back(root);
}

categoryTree::~categoryTree()
{
  delete root; // Will recursively delete all children
}

void categoryTree::print(void)
{
  root->print(0);
}

void categoryTree::fprintJson(FILE* f)
{
  root->fprintJson(0, f);
}

/// Add a new category to the category tree, whose name is given by a vector of strings
categoryNode* categoryTree::addPath(vector<string> category)
{
  string* categoryP = &(category[0]);
  categoryNode* deepest = root;
  categoryNode* child = 0;
  int L = category.size();

  if (skipRoot)
  {
    categoryP ++;
    L --;
    if (L == 0) return root;
  }

  while (L > 0 and (child = deepest->find(categoryP, 1)))
  {
    categoryP ++;
    deepest = child;
    L --;
  }
  // We ran out of children. Need to add the rest.
  while (L > 0)
  {
    int nextId = (int) idToNode.size();

    child = new categoryNode(categoryP[0], deepest, nextId, productsPerTopic);
    deepest->addChild(child);
    deepest = child;
    categoryP ++;
    L --;

    // Give each new child a new id. This code should be changed if we only want to give leaf nodes ids.
    nodeToId[child] = nextId;
    idToNode.push_back(child);
  }

  return child;
}

/// From a leaf node (or any node really) get the path of nodes above it going back to the root
vector<categoryNode*> categoryTree::pathFromId(int nodeId)
{
  vector<categoryNode*> pathR;

  categoryNode* current = idToNode[nodeId];
  while (current)
  {
    pathR.push_back(current);
    current = current->parent;
  }
  reverse(pathR.begin(), pathR.end());
  return pathR;
}

/// Increment all nodes along a path (e.g. for a product in Electronics->Mobile Phones->Accessories increment all three category nodes)
void categoryTree::incrementCounts(int nodeId)
{
  categoryNode* current = idToNode[nodeId];
  while (current)
  {
    current->observe();
    current = current->parent;
  }
}

int categoryTree::nNodes(void)
{
  return idToNode.size();
}

/// Associate each node in the category tree with some topics
topicTree::topicTree(categoryTree* ct) : ct(ct)
{
  totalTopics = 0;
  for (int i = 0; i < ct->nNodes(); i ++)
  {
    int howMany = ct->idToNode[i]->howManyTopics(); // How many topics should be associated with this node?
    topicStart.push_back(totalTopics);
    topicCount.push_back(howMany);
    totalTopics += howMany;
    for (int j = 0; j < howMany; j ++)
      topicNode.push_back(i);
  }

  // Associate each topic with *one of* the topics from its parent category
  // Could be used for regularizaton but is currently unused
  for (int i = 0; i < ct->nNodes(); i ++)
  {
    categoryNode* n = ct->idToNode[i];
    categoryNode* parent = n->parent;
    if (parent)
      for (int j = 0; j < topicCount[i]; j ++)
      {
        // Only valid because the parent always has more topics than the child.
        topicParent[topicStart[n->nodeId] + j] = topicStart[parent->nodeId] + j;
      }
  }
}
