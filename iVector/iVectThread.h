#ifndef IVECTTHREAD_H
#define IVECTTHREAD_H
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include "iVectMath.h"
#include <vector>
#include <iostream>

void updateiVectors(std::vector<Document> &documents, FeatureSpace &space, int numOfThreads);
void updatetRows(std::vector<Document> &documents, FeatureSpace & space, int numOfThreads);
void updatetRows(std::vector<Document> &documents, FeatureSpace & space, int numOfThreads, std::vector<Document> & devDocs);

#endif