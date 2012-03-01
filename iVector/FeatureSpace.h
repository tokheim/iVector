#ifndef FEATURESPACE_H
#define FEATURESPACE_H

#include <cstdlib>
#include "Document.h"
#include <math.h>
using namespace std;

struct FeatureSpace {
	vector<double> mVector;
	vector< vector<double> > tMatrix;
	vector< vector<double> > oldtMatrix;
	unsigned int height;
	unsigned int width;

	FeatureSpace(unsigned int height, unsigned int width, vector<Document> & documents, unsigned int seed);
	FeatureSpace(vector< vector<double> > tMatrix, vector<Document> & documents);
	FeatureSpace(vector< vector<double> > tMatrix, vector<double> mVector);
	void generatetMatrix(unsigned int seed);
	void generatemVector(vector<Document> & documents);
};
#endif