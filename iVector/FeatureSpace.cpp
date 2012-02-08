#include "FeatureSpace.h"



//const static unsigned int SEED = 23;

FeatureSpace::FeatureSpace(int height, int width, vector<Document> & documents, unsigned int seed) {
	this->height = height;
	this->width = width;
	generatetMatrix(seed);
	oldtMatrix = new double * [height];
	generatemVector(documents);
}
FeatureSpace::FeatureSpace(int height, int width, double ** tMatrix, vector<Document> & documents) {
	this->height = height;
	this->width = width;
	this->tMatrix = tMatrix;
	this->oldtMatrix = new double * [height];
	generatemVector(documents);
}

void FeatureSpace::generatetMatrix(unsigned int seed) {
	srand(seed);
	tMatrix = new double * [height];
	for (int i = 0; i < height; i++) {
		tMatrix[i] = new double[width];
		for (int j = 0; j < width; j++) {
			tMatrix[i][j] = (((double) rand())/RAND_MAX-0.5);
		}
	}
}
//Endre iterator
void FeatureSpace::generatemVector(vector<Document> & documents) {
	mVector = new double[height];
	for (int i = 0; i < height; i++) {
		mVector[i] = 0;
	}
	HASH_I_D::iterator it;
	for (unsigned int i = 0; i < documents.size(); i++) {
		for (it = documents[i].gamma.begin(); it != documents[i].gamma.end(); ++it) {
			mVector[it->first] += it->second;
		}
	}
	for (int i = 0; i < height; i++) {
		mVector[i] = log(mVector[i]/documents.size());
	}
}