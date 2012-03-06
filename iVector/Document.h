#ifndef DOCUMENT_H
#define DOCUMENT_H
#include <string>
#include <float.h>
#include <boost/numeric/ublas/vector.hpp>
#include <vector>

//OS specific (windows or unix-based assumed)
#ifdef _WIN32
#include <hash_map>
typedef stdext::hash_map<int, double> HASH_I_D;
#else
#include <ext/hash_map>
typedef __gnu_cxx::hash_map<int, double> HASH_I_D;
#endif

typedef std::pair <int, double> I_D_PAIR;

struct Document {
	int languageClass;
	boost::numeric::ublas::vector<double> iVector;
	boost::numeric::ublas::vector<double> oldiVector;
	double gammaSum;
	double lastLikelihood;
	HASH_I_D gamma;
	Document (int languageClass, HASH_I_D & gamma, int dim);
    Document ();
	void calcGammaSum();
	void setupIvectors(int dim);
	double getGammaValue(int feature);
};
void useOldiVectors(std::vector<Document> &documents);
void resetiVectors(std::vector<Document> &documents);
#endif