#ifndef IVECTMATH_H
#define IVECTMATH_H
#include <math.h>
#include "Document.h"
#include "FeatureSpace.h"
#include <iostream>
#include "log.h"

using namespace std;

double * initializeVector(int length);
double ** initializeMatrix(int height, int width);
double * addVectors(double * vectora, double * vectorb, int length);
double multiplyVectors(double * rowvector, double * columnvector, int length);
double * scaleVector(double * vector, double scalar, int length);
void scaleAndAddVector(double * addToVector, double * vector, double scalar, int length);
double eucledianDistance(double * vectora, double * vectorb, int length);
double * avgVector(double * vectorA, double * vectorB, int dim);
int * lupDecompose(double ** a, int dim);
double * lupSolve(double ** LU, double * b, int * p, int dim);
void updateiVector(Document & document, FeatureSpace & space);

double calcTotalLikelihood(vector<Document> & documents, FeatureSpace & space);
double calcTotalLikelihoodExcludeInf(vector<Document> & documents, FeatureSpace & space);
void updateiVectors(vector<Document> & documents, FeatureSpace & space);
void updatetRows(vector<Document> & documents, FeatureSpace & space);
void updateiVectorsCheckStep(vector<Document> & documents, FeatureSpace & space);
bool updatetRowsCheckStep(vector<Document> & documents, FeatureSpace & space);

#endif