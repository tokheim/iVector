#include "Document.h"
#include "iVectMath.h"



Document::Document(int languageClass, HASH_I_D & gamma, int dim) {
	this->languageClass = languageClass;
	lastLikelihood = -DBL_MAX;
	this->gamma = gamma;
	calcGammaSum();
	setupIvectors(dim);
}

void Document::setupIvectors(int dim) {
	iVector.clear();
	iVector.resize(dim, 0.0);
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
void resetiVectors(vector<Document> &documents) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		documents[i].setupIvectors(documents[0].iVector.size());
	}
}
