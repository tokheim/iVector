#include "FeatureSpace.h"
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

/*
This class holds the feature space that is used to extract iVectors. This includes the median vector (mVector) and the
total variability matrix (tMatrix).  There are also methods to initialize these parameters
*/

FeatureSpace::FeatureSpace(unsigned int height, unsigned int width, std::vector<Document> & documents, unsigned int seed) {
	this->height = height;
	this->width = width;
	generatetMatrix(seed);
	generatemVector(documents);
	//scaleSpaceByVar(documents);
}
FeatureSpace::FeatureSpace(matrix<double> tMatrix, std::vector<Document> & documents) {
	height = tMatrix.size1();
	width = tMatrix.size2();
	this->tMatrix = tMatrix;
	this->oldtMatrix = tMatrix;
	generatemVector(documents);
}
FeatureSpace::FeatureSpace(matrix<double> tMatrix, vector<double> mVector) {
	height = tMatrix.size1();
	width = tMatrix.size2();
	this->tMatrix = tMatrix;
	this->oldtMatrix = tMatrix;
	this->mVector = mVector;
}


void FeatureSpace::generatetMatrix(unsigned int seed) {
	srand(seed);
	tMatrix.resize(height, width, false);
	for (unsigned int i = 0; i < height; i++) {
		for (unsigned int j = 0; j < width; j++) {
			tMatrix(i, j) = (((double) rand())/RAND_MAX-0.5);
		}
	}
	oldtMatrix = tMatrix;
}

void FeatureSpace::generatemVector(std::vector<Document> & documents) {
	mVector.resize(height, false);
	mVector.clear();
	HASH_I_D::iterator it;
	for (unsigned int i = 0; i < documents.size(); i++) {
		for (it = documents[i].gamma.begin(); it != documents[i].gamma.end(); ++it) {
			mVector(it->first) += it->second;
		}
	}
	for (unsigned int i = 0; i < height; i++) {
		mVector(i) = log(mVector(i)/documents.size());
	}
}

void FeatureSpace::scaleSpaceByVar(std::vector<Document> & documents) {
	for (unsigned int i = 0; i < height; i++) {
		double avg = exp(mVector(i));
		if (avg != 0.0) {
			double var = 0;
			for (unsigned int j = 0; j < documents.size(); j++) {
				var += pow(avg-documents[j].getGammaValue(i), 2);
			}
			row(tMatrix, i) = (sqrt(var/(documents.size()-1))/width) * row(tMatrix, i);
		}
	}
}

