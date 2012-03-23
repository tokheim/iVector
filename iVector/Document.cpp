#include "Document.h"
#include "iVectMath.h"
/*
This class holds the information for a spoken document. This includes the original feature vector (gamma), the language class
and the document's iVector. There is two iVectors so that the iterative algorithm can revert to the previous iVector.
*/

Document::Document(int languageClass, HASH_I_D & gamma, int dim) {
	this->languageClass = languageClass;
	lastLikelihood = -DBL_MAX;
	this->gamma = gamma;
	calcGammaSum();
	setupIvectors(dim);
}
Document::Document() {
    
}

void Document::setupIvectors(int dim) {
	iVector.resize(dim, false);
	oldiVector.resize(dim, false);
	iVector.clear();
	oldiVector.clear();
}

//Precalculates the first norm of the original feature vector
void Document::calcGammaSum() {
	gammaSum = 0;
	HASH_I_D::iterator it;
	for (it = gamma.begin(); it != gamma.end(); ++it) {
		gammaSum += it->second;
	}
}
//Returns the value for the original feature (or zero if it doesn't exist in the sparse representation)
const double Document::getGammaValue(int feature) {
	HASH_I_D::const_iterator it = gamma.find(feature);
	//HASH_I_D::iterator it = gamma.find(feature);
	if (it != gamma.end()) {
		return it->second;
	} else {
		return 0.0;
	}
}
void useOldiVectors(std::vector<Document> &documents) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		documents[i].iVector = documents[i].oldiVector;
	}
}
void resetiVectors(std::vector<Document> &documents) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		documents[i].setupIvectors(documents[0].iVector.size());
	}
}
