#include "FeatureSpace.h"



//const static unsigned int SEED = 23;

FeatureSpace::FeatureSpace(unsigned int height, unsigned int width, vector<Document> & documents, unsigned int seed) {
	this->height = height;
	this->width = width;
	generatetMatrix(seed);
	generatemVector(documents);
}
FeatureSpace::FeatureSpace(vector< vector<double> > tMatrix, vector<Document> & documents) {
	height = tMatrix.size();
	width = tMatrix[0].size();
	this->tMatrix = tMatrix;
	this->oldtMatrix = tMatrix;
	generatemVector(documents);
}
FeatureSpace::FeatureSpace(vector< vector<double> > tMatrix, vector<double> mVector) {
	height = tMatrix.size();
	width = tMatrix[0].size();
	this->tMatrix = tMatrix;
	this->oldtMatrix = tMatrix;
	this->mVector = mVector;
}


void FeatureSpace::generatetMatrix(unsigned int seed) {
	srand(seed);
	tMatrix.resize(height);
	for (unsigned int i = 0; i < height; i++) {
		tMatrix[i].resize(width);
		for (unsigned int j = 0; j < width; j++) {
			tMatrix[i][j] = (((double) rand())/RAND_MAX-0.5);
		}
	}
	oldtMatrix = tMatrix;//Should copy the values
}
//Endre iterator
void FeatureSpace::generatemVector(vector<Document> & documents) {
	mVector.resize(height, 0.0);
	HASH_I_D::iterator it;
	for (unsigned int i = 0; i < documents.size(); i++) {
		for (it = documents[i].gamma.begin(); it != documents[i].gamma.end(); ++it) {
			mVector[it->first] += it->second;
		}
	}
	for (unsigned int i = 0; i < height; i++) {
		mVector[i] = log(mVector[i]/documents.size());
	}
}