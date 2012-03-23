#include "iVectMath.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <math.h>

/*
This class does most of the mathematical operations in the program
*/

const static int MAX_REDUCE_STEPSIZE_ATTEMPTS = 4;
const static double MINUS_INF = log(0.0);

using namespace boost::numeric::ublas;



double calcAvgEuclideanDistance(std::vector<Document> & documents) {
	double dist = 0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		dist += norm_2(documents[i].iVector - documents[i].oldiVector);
	}
	return dist/documents.size();
}

//Calculates the denominator in the expression for phi for a document
double calcPhiDenominator(FeatureSpace & space, vector<double> & iVector) {
	double denominator = 0.0;
	vector<double> prods = prod(space.tMatrix, iVector);
	for (unsigned int i = 0; i < space.height; i++) {
		denominator += exp(space.mVector(i)+prods(i));
		//denominator += exp(space.mVector(i)+inner_prod(space.tMatrix(i), iVector));Alternative
	}
	return denominator;
}

//Calculates the denominator in phi for a set of documents. Usefull precalculation when updating the rows of T
vector<double> calcAllPhiDenominators(FeatureSpace & space, std::vector<Document> & documents) {
	vector<double> denominators(documents.size());
	for (unsigned int i = 0; i < documents.size(); i++) {
		denominators(i) = calcPhiDenominator(space, documents[i].iVector);
	}
	return denominators;
}

double calcPhi(FeatureSpace & space, vector<double> & iVector, unsigned int row, double denominator) {
	return exp(space.mVector(row)+inner_prod(matrix_row<matrix<double> >(space.tMatrix, row), iVector))/denominator;
}

double calcUtteranceLikelihood(Document & document, FeatureSpace & space) {
	double likelihood = 0.0;
	double denominator = calcPhiDenominator(space, document.iVector);
	HASH_I_D::iterator it;
	for (it = document.gamma.begin(); it != document.gamma.end(); ++it) {
		double phi = calcPhi(space, document.iVector, it->first, denominator);
		likelihood += it->second * log(phi);
	}
	return likelihood;
}

double calcTotalLikelihood(std::vector<Document> & documents, FeatureSpace & space) {
	double likelihood = 0.0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		likelihood += calcUtteranceLikelihood(documents[i], space);
	}
	return likelihood;
}
//This gives the likelihood but excludes any feature that has zero probability of occuring. These features should be constant
//throughout iterations.
double calcUtteranceLikelihoodExcludeInf(Document & document, FeatureSpace & space) {
	double likelihood = 0.0;
	double denominator = calcPhiDenominator(space, document.iVector);
	HASH_I_D::iterator it;
	for (it = document.gamma.begin(); it != document.gamma.end(); ++it) {
		if (space.mVector(it->first) != MINUS_INF) {
			likelihood += it->second * log(calcPhi(space, document.iVector, it->first, denominator));
		}
	}
	return likelihood;
}

double calcTotalLikelihoodExcludeInf(std::vector<Document> & documents, FeatureSpace & space) {
	double totLikelihood = 0.0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		totLikelihood += calcUtteranceLikelihoodExcludeInf(documents[i], space);
	}
	return totLikelihood;
}
//setup Ax=b for iVector updates
void setUpSystem(vector<double> & gradient, symmetric_matrix<double> & jacobian, Document & document, FeatureSpace & space, double denominator) {
	jacobian.clear();
	gradient.clear();
	for (unsigned int trow = 0; trow < space.height; trow++) {
		double gammaVal = document.getGammaValue(trow);
		if (space.mVector(trow) == MINUS_INF) {//If feature is not observed in training set, then don't model it (Assume the row in T is all zero)
			continue;
		}
		double phiPart = calcPhi(space, document.iVector, trow, denominator)*document.gammaSum;
		double gradweight = gammaVal-phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		vector<double> tRow = row(space.tMatrix, trow);//evt. bruke row(space.tMatrix, trow)
		noalias(gradient) += gradweight * tRow;
		noalias(jacobian) += jacweight*outer_prod(tRow, tRow);
	}
}
//Setup Ax=b for rows of T
void setUpSystem(vector<double> & gradient, symmetric_matrix<double> & jacobian, std::vector<Document> & documents, FeatureSpace & space, unsigned int row, vector<double> & denominators) {
	jacobian.clear();
	gradient.clear();
	for (unsigned int n = 0; n < documents.size(); n++) {
		double gammaVal = documents[n].getGammaValue(row);
		double phiPart = calcPhi(space, documents[n].iVector, row, denominators(n))*documents[n].gammaSum;
		double gradweight = gammaVal - phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		noalias(gradient) += gradweight*documents[n].iVector;
		noalias(jacobian) += jacweight*outer_prod(documents[n].iVector, documents[n].iVector);
	}
}
//Perform a newton raphson update step on a iVector
void updateiVector(Document & document, FeatureSpace & space) {
	document.oldiVector = document.iVector;
	double denominator = calcPhiDenominator(space, document.iVector);
	vector<double> b(space.width);
	symmetric_matrix<double> jacobian(space.width);
	setUpSystem(b, jacobian, document, space, denominator);
	
	matrix<double> A = jacobian;//Neccessary since the jacobian is symmetric while the decomposed matrix is not

	permutation_matrix<double> p(space.width);
	lu_factorize(A, p);
	lu_substitute(A, p, b);//b holds the solution
	document.iVector += b;
}
//Perform newton Raphson updates on all iVectors in set
void updateiVectors(std::vector<Document> & documents, FeatureSpace & space) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		updateiVector(documents[i], space);
	}
}
//Performs newton raphson updates on a row of T
void updatetRow(std::vector<Document> & documents, FeatureSpace & space, unsigned int row, vector<double> & denominators) {
	matrix_row<matrix<double> > (space.oldtMatrix, row) = matrix_row<matrix<double> > (space.tMatrix, row);
	if (space.mVector(row) != MINUS_INF) {//If mVector is trained on a superset of "documents" then the old values should still be a solution
		vector<double> b(space.width);
		symmetric_matrix<double> jacobian(space.width);
		setUpSystem(b, jacobian, documents, space, row, denominators);

		matrix<double> A = jacobian;//Necccessary since the jacobian is symmetric while the decomposed matrix is not

		permutation_matrix<double> p(space.width);
		lu_factorize(A, p);
		lu_substitute(A, p, b);//b holds solution
		matrix_row<matrix<double> > (space.tMatrix, row) += b;
	}
}
//Performs newton-raphson updates on all rows of T
void updatetRows(std::vector<Document> & documents, FeatureSpace & space) {
	vector<double> denominators = calcAllPhiDenominators(space, documents);
	for (unsigned int row = 0; row < space.height; row++) {
		updatetRow(documents, space, row, denominators);
	}
}

//Recursivly reduce the update step until the likelihood increases
void recursiveiVectorUpdateCheck(Document & document, FeatureSpace & space, double oldLikelihood, int attempts) {
	if (oldLikelihood > calcUtteranceLikelihoodExcludeInf(document, space) && attempts < MAX_REDUCE_STEPSIZE_ATTEMPTS) {
		document.iVector = (document.iVector-document.oldiVector)/2;
		recursiveiVectorUpdateCheck(document, space, oldLikelihood, attempts+1);
	}
	else if (attempts >= MAX_REDUCE_STEPSIZE_ATTEMPTS) {//Could not find a better iVector, so use the old one
		document.iVector = document.oldiVector;
	}
	//Else: the set iVector is better than the old one, do nothing
}
//Ensure that the update step of an iVector doesn't cause the likelihood to decrease by reducing the update steps
void updateiVectorCheckLike(Document & document, FeatureSpace & space) {
	double oldLikelihood = calcUtteranceLikelihoodExcludeInf(document, space);//could be changed from T-matrix updates
	updateiVector(document, space);
	recursiveiVectorUpdateCheck(document, space, oldLikelihood, 0);
}