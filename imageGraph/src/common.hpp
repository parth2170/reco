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

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include "vector"
#include "common.hpp"
#include "algorithm"
#include "sstream"
#include "gzstream.h"

enum action_t { INIT, COPY, FREE };
enum error_t { TRAIN, VALID, TEST };

/// Safely open a file
FILE* fopen_(const char* p, const char* m);

double clock_(void);

double inner(std::map<int, double>* x, double* w);
double inner(double* x, double* y, int K);

double sigmoid(double x);
double square(double x);

double safeLog(double p);

class edge
{
public:
  edge(int productFrom,
       int productTo,
       int label, // Is this an edge or a non-edge
       int reverseLabel); // Does the graph have an edge going the other direction?

  ~edge();

  int productFrom;
  int productTo;

  int label;
  int reverseLabel;
};

/// Data associated with a rating
struct vote
{
  int user; // ID of the user
  int item; // ID of the item
  float value; // Rating

  int voteTime; // Unix time of the rating
  std::vector<int> words; // IDs of the words in the review
};

typedef struct vote vote;
