#ifndef IVECTMATH_H
#define IVECTMATH_H
#include <math.h>
#include "Document.h"
#include "FeatureSpace.h"
#include <iostream>

using namespace std;

void addVectors(vector<double> &addToVector, vector<double> &addFromVector);
double multiplyVectors(vector<double> &rowvector, vector<double> &columnvector);
void scaleAndAddVector(vector<double> &addToVector, vector<double> &scalevector, double scalar);
double eucledianDistance(vector<double> &vectora, vector<double> &vectorb);
void avgVector(vector<double> &saveToVector, vector<double> &unchangedVector);
double calcPhiDenominator(FeatureSpace & space, vector<double> &iVector);
vector<double> calcAllPhiDenominators(FeatureSpace & space, vector<Document> & documents);
double calcPhi(FeatureSpace & space, vector<double> &iVector, int row, double denominator);

void setUpSystem(vector<double> & b, vector< vector<double> > & jacobian, Document & document, FeatureSpace & space, double denominator);
void setUpSystem(vector<double> &b, vector< vector<double> > & jacobian, vector<Document> & documents, FeatureSpace & space, int row, vector<double> & denominators);
void lupDecompose(vector< vector<double> > & a);
vector<double> lupSolve(vector< vector<double> > &LU, vector<double> & b);

double calcUtteranceLikelihood(Document & document, FeatureSpace & space);
double calcTotalLikelihood(vector<Document> & documents, FeatureSpace & space);
double calcUtteranceLikelihoodExcludeInf(Document & document, FeatureSpace & space);
double calcTotalLikelihoodExcludeInf(vector<Document> & documents, FeatureSpace & space);
void updateiVector(Document & document, FeatureSpace & space);
void updatetRow(vector<Document> & documents, FeatureSpace & space, int row, vector<double> & denominators);

void updateiVectors(vector<Document> & documents, FeatureSpace & space);
void updatetRows(vector<Document> & documents, FeatureSpace & space);
#endif