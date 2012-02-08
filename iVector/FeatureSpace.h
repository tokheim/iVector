#ifndef FEATURESPACE_H
#define FEATURESPACE_H

#include <cstdlib>
#include "Document.h"
#include <math.h>
using namespace std;

struct FeatureSpace {
	double * mVector;
	double ** tMatrix;
	double ** oldtMatrix;
	int height;
	int width;

	FeatureSpace(int height, int width, vector<Document> & documents, unsigned int seed);
	FeatureSpace(int height, int width, double ** tMatrix, vector<Document> & documents);
	void generatetMatrix(unsigned int seed);
	void generatemVector(vector<Document> & documents);
};
#endif