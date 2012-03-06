#ifndef FEATURESPACE_H
#define FEATURESPACE_H

#include <cstdlib>
#include "Document.h"
#include <math.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>


struct FeatureSpace {
	boost::numeric::ublas::vector<double> mVector;
	boost::numeric::ublas::matrix<double> tMatrix;
	boost::numeric::ublas::matrix<double> oldtMatrix;
	unsigned int height;
	unsigned int width;

	FeatureSpace(unsigned int height, unsigned int width, std::vector<Document> & documents, unsigned int seed);
	FeatureSpace(boost::numeric::ublas::matrix<double> tMatrix, std::vector<Document> & documents);
	FeatureSpace(boost::numeric::ublas::matrix<double> tMatrix, boost::numeric::ublas::vector<double> mVector);
	void generatetMatrix(unsigned int seed);
	void generatemVector(std::vector<Document> & documents);
};
#endif