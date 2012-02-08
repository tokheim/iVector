#ifndef DOCUMENT_H
#define DOCUMENT_H
#include <string>
#include <float.h>


//OS specific (windows or unix-based assumed)
#ifdef _WIN32
#include <hash_map>
typedef stdext::hash_map<int, double> HASH_I_D;
#else
#include <ext/hash_map>
typedef __gnu_cxx::hash_map<int, double> HASH_I_D;
#endif

using namespace std;

struct Document {
	int languageClass;
	double * iVector;
	double * oldiVector;
	double gammaSum;
	double lastLikelihood;
	HASH_I_D gamma;
	Document (int languageClass, HASH_I_D & gamma, int dim);
	void calcGammaSum();
	void setupIvectors(int dim);
	double getGammaValue(int feature);
};
void useOldiVectors(vector<Document> &documents);
void resetiVectors(vector<Document> &documents, int dim);
#endif