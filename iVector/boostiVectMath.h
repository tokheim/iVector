#ifndef IVECTMATH_H
#define IVECTMATH_H
#include <math.h>
#include "Document.h"
#include "FeatureSpace.h"

double calcAvgEuclideanDistance(std::vector<Document> & documents);
double calcTotalLikelihood(std::vector<Document> & documents, FeatureSpace & space);
double calcTotalLikelihoodExcludeInf(std::vector<Document> & documents, FeatureSpace & space);
void updateiVectors(std::vector<Document> & documents, FeatureSpace & space);
void updatetRows(std::vector<Document> & documents, FeatureSpace & space);


#endif
