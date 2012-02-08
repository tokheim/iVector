#include "Document.h"
#include "iVectMath.h"

Document::Document(int languageClass, HASH_I_D & gamma, int dim) {
	this->languageClass = languageClass;
	lastLikelihood = -DBL_MAX;
	this->gamma = gamma;
	calcGammaSum();
	setupIvectors(dim);
	//lastLikelihood = - numeric_limits<double>::max()
}

void Document::setupIvectors(int dim) {
	iVector = initializeVector(dim);
	oldiVector = iVector;
}

void Document::calcGammaSum() {
	gammaSum = 0;
	HASH_I_D::iterator it;
	for (it = gamma.begin(); it != gamma.end(); ++it) {
		gammaSum += it->second;
	}
}
double Document::getGammaValue(int feature) {
	HASH_I_D::iterator it = gamma.find(feature);
	if (it != gamma.end()) {
		return it->second;
	} else {
		return 0.0;
	}
}
void useOldiVectors(vector<Document> &documents) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		documents[i].iVector = documents[i].oldiVector;
	}
}
void resetiVectors(vector<Document> &documents, int dim) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		documents[i].iVector = initializeVector(dim);
		documents[i].oldiVector = documents[i].iVector;
	}
}
