#ifndef IVECTMATH_H
#define IVECTMATH_H
#include <math.h>
#include "Document.h"
#include "FeatureSpace.h"
#include <boost/numeric/ublas/vector.hpp>
#include <vector>

void colOrthogonalize(boost::numeric::ublas::matrix<double > & mat);
boost::numeric::ublas::vector<double> calcAllPhiDenominators(FeatureSpace & space, std::vector<Document> & documents);
double calcAvgEuclideanDistance(std::vector<Document> & documents);
double calcTotalLikelihood(std::vector<Document> & documents, FeatureSpace & space, bool excludeInf);

void updateiVectors(std::vector<Document> & documents, FeatureSpace & space);
void updateiVectorCheckLike(Document & document, FeatureSpace & space);
void updateiVector(Document & document, FeatureSpace & space);

void updatetRows(std::vector<Document> & documents, FeatureSpace & space);
void updatetRow(std::vector<Document> & documents, FeatureSpace & space, unsigned int row, boost::numeric::ublas::vector<double> & denominators);


//Experimental


void updatetRowCheckLike(std::vector<Document> & documents, FeatureSpace & space, unsigned int tRow, boost::numeric::ublas::vector<double> & denominators);
void checkTLike(std::vector<Document> & documents, FeatureSpace & space, unsigned int column);//checks in columns
void updatetRowCheckLike(std::vector<Document> & documents, FeatureSpace & space, unsigned int tRow, boost::numeric::ublas::vector<double> & denominators, std::vector<Document> & devDocs);

void updatetRowPart(std::vector<Document> & documents, FeatureSpace & space, unsigned int row, boost::numeric::ublas::vector<double> & denominators);

//Experimental testing
//double getColumnLikelihood(std::vector<Document> & documents, boost::numeric::ublas::matrix<double> & tMatrix, boost::numeric::ublas::vector<double> & mVector, unsigned int column);
boost::numeric::ublas::vector<double> calcPhiNominators(std::vector<Document> & documents, FeatureSpace & space, int tRow);
double calcLikelihood(std::vector<Document> & documents, FeatureSpace & space, boost::numeric::ublas::vector<double> & denominators, boost::numeric::ublas::vector<double> & oldNominators, unsigned int tRow);

#endif
