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

void colOrthogonalize(matrix<double > & mat) {
	for (unsigned int i = 0; i < mat.size2(); i++) {
		double prod = inner_prod(column(mat, i), column(mat, i));
		vector<double> u = column(mat, i)/prod;
		for (unsigned int j = i+1; j < mat.size2(); j++) {
			column(mat, j) -= inner_prod(column(mat, i), column(mat, j))*u;
		}
	}
}


//Calculates the denominator in the expression for phi for a document
double calcPhiDenominator(FeatureSpace & space, vector<double> & iVector) {
	double denominator = 0.0;
	vector<double> prods = prod(space.tMatrix, iVector);
	for (unsigned int i = 0; i < space.height; i++) {
		denominator += exp(space.mVector(i)+prods(i));
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

double calcUtteranceLikelihood(Document & document, FeatureSpace & space, bool excludeInf) {
	double likelihood = 0.0;
	double denominator = calcPhiDenominator(space, document.iVector);
	HASH_I_D::iterator it;
	if (!excludeInf) {
		for (it = document.gamma.begin(); it != document.gamma.end(); ++it) {
			likelihood += it->second * log(calcPhi(space, document.iVector, it->first, denominator));
		}
	}
	else {
		for (it = document.gamma.begin(); it != document.gamma.end(); ++it) {
			if (space.mVector(it->first) != MINUS_INF) {
				likelihood += it->second * log(calcPhi(space, document.iVector, it->first, denominator));
			}
		}
	}
	return likelihood;
}

double calcTotalLikelihood(std::vector<Document> & documents, FeatureSpace & space, bool excludeInf) {
	double likelihood = 0.0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		likelihood += calcUtteranceLikelihood(documents[i], space, excludeInf);
	}
	return likelihood;
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
	lu_factorize(A, p);//LU-decomposes A
	lu_substitute(A, p, b);//b holds the solution (change in given iVector)
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
	if (space.mVector(row) != MINUS_INF) {//If mVector is trained on a superset of "documents" then the old values should still be a solution optimal solution
		vector<double> b(space.width);
		symmetric_matrix<double> jacobian(space.width);
		setUpSystem(b, jacobian, documents, space, row, denominators);

		matrix<double> A = jacobian;//Neccessary since the jacobian is symmetric while the decomposed matrix is not

		permutation_matrix<double> p(space.width);
		lu_factorize(A, p);
		lu_substitute(A, p, b);//b holds the solution (change in given row of T)
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
	if (oldLikelihood > calcUtteranceLikelihood(document, space, true) && attempts < MAX_REDUCE_STEPSIZE_ATTEMPTS) {
		document.iVector = (document.iVector+document.oldiVector)/2;
		recursiveiVectorUpdateCheck(document, space, oldLikelihood, attempts+1);
	}
	else if (attempts >= MAX_REDUCE_STEPSIZE_ATTEMPTS) {//Could not find a better iVector, so use the old one
		document.iVector = document.oldiVector;
	}
	//Else: the set iVector is better than the old one, do nothing
}
//Ensure that the update step of an iVector doesn't cause the likelihood to decrease by reducing the update steps
void updateiVectorCheckLike(Document & document, FeatureSpace & space) {
	double oldLikelihood = calcUtteranceLikelihood(document, space, true);//could be changed from T-matrix updates
	updateiVector(document, space);
	recursiveiVectorUpdateCheck(document, space, oldLikelihood, 0);
}





/*
Experimental code, after t-update checks that each column recieves higher likelihood (by itself)
*/



double getColumnLikelihood(std::vector<Document> & documents, matrix<double> & tMatrix, vector<double> & mVector, unsigned int column) {
	double totLike = 0.0;
	vector<double> nominators(mVector.size());
	double denom;
	HASH_I_D::iterator it;
	for (unsigned int i = 0; i < documents.size(); i++) {
		denom = 0.0;
		for (unsigned int j = 0; j < mVector.size(); j++) {
			nominators(j) = exp(mVector(j)+tMatrix(j, column)*documents[i].iVector(column));
			//denom += nominators(j);
			denom += exp(mVector(j)+inner_prod(row(tMatrix, j), documents[i].iVector));
		}
		for (it = documents[i].gamma.begin(); it != documents[i].gamma.end(); ++it) {
			if (mVector(it->first) != MINUS_INF) {
				totLike += it->second * log(nominators(it->first)/denom);
			}
		}
	}
	return totLike;
}

void checkTLike(std::vector<Document> & documents, FeatureSpace & space, double oldLikelihood, unsigned int tColumn, int attempts) {
	if (attempts >= MAX_REDUCE_STEPSIZE_ATTEMPTS) {//Could not find better row, so use old one
		column(space.tMatrix, tColumn) = column(space.oldtMatrix, tColumn);
	}
	else {
		if (oldLikelihood >= getColumnLikelihood(documents, space.tMatrix, space.mVector, tColumn)) {
			column(space.tMatrix, tColumn) = (column(space.oldtMatrix, tColumn)+column(space.tMatrix, tColumn))/2;
			checkTLike(documents, space, oldLikelihood, tColumn, attempts+1);
		}
	}
}


void checkTLike(std::vector<Document> & documents, FeatureSpace & space, unsigned int column) {
	double oldLikelihood = getColumnLikelihood(documents, space.oldtMatrix, space.mVector, column);
	checkTLike(documents, space, oldLikelihood, column, 0);
}



/*
Experimental code, updates row of t and check that total likelihood is greater after update
*/




//Eksperimentell kode
vector<double> calcPhiNominators(std::vector<Document> & documents, FeatureSpace & space, int tRow) {
	vector<double> nominators(documents.size());
	for (unsigned int i = 0; i < documents.size(); i++) {
		nominators(i) = exp(space.mVector(tRow)+inner_prod(documents[i].iVector, row(space.tMatrix, tRow)));
	}
	return nominators;
}

double calcLikelihood(std::vector<Document> & documents, FeatureSpace & space, vector<double> & denominators, vector<double> & oldNominators, unsigned int tRow) {
	double likeDiff = 0.0;
	double nominator;
	double rowGamma;
	for (unsigned int i = 0; i < documents.size(); i++) {
		nominator = exp(space.mVector(tRow)+inner_prod(row(space.tMatrix, tRow), documents[i].iVector));
		rowGamma = documents[i].getGammaValue(tRow);
		likeDiff += rowGamma * (log(nominator/(denominators(i)+nominator-oldNominators(i)))-log(oldNominators(i)/denominators(i)));
		likeDiff += (documents[i].gammaSum-rowGamma)*(log(denominators(i))-log(denominators(i)+nominator-oldNominators(i)));
	}
	return likeDiff;
}

void recursivetRowUpdateCheck(std::vector<Document> & documents, FeatureSpace & space, vector<double> & denominators, vector<double> & oldNominators, unsigned int tRow, int attempts) {
	if (attempts >= MAX_REDUCE_STEPSIZE_ATTEMPTS) {//Could not find better row, so use old one
		row(space.tMatrix, tRow) = row(space.oldtMatrix, tRow);
	}
	else {
		if (0 > calcLikelihood(documents, space, denominators, oldNominators, tRow)) {
			row(space.tMatrix, tRow) = (row(space.oldtMatrix, tRow)+row(space.tMatrix, tRow))/2;
			recursivetRowUpdateCheck(documents, space, denominators, oldNominators, tRow, attempts+1);
		}
	}
}


void updatetRowCheckLike(std::vector<Document> & documents, FeatureSpace & space, unsigned int tRow, vector<double> & denominators) {
	if (space.mVector(tRow) != MINUS_INF) {
		vector<double> oldNominators = calcPhiNominators(documents, space, tRow);
		updatetRow(documents, space, tRow, denominators);
		recursivetRowUpdateCheck(documents, space, denominators, oldNominators, tRow, 0);
	}
}




/*
Experimental code, Updates only part of a row of T
*/

//Updates part of a row of T usefull for resizing
void updatetRowPart(std::vector<Document> & documents, FeatureSpace & space, unsigned int row, vector<double> & denominators) {
	unsigned int fromIndex = space.width-50;//Is hardcoded for testing, should be separate
	matrix_row<matrix<double> > (space.oldtMatrix, row) = matrix_row<matrix<double> > (space.tMatrix, row);
	if (space.mVector(row) != MINUS_INF) {//If mVector is trained on a superset of "documents" then the old values should still be a solution optimal solution
		vector<double> grad(space.width);
		symmetric_matrix<double> jacobian(space.width);
		setUpSystem(grad, jacobian, documents, space, row, denominators);

		//Select part of gradient/jacboian that should be updated
		vector<double> b(space.width-fromIndex);
		matrix<double> A(space.width-fromIndex, space.width-fromIndex);
		for (unsigned int i = fromIndex; i < space.width; i++) {
			b(i-fromIndex) = grad(i);
			for (unsigned int j = fromIndex; j < space.width; j++) {
				A(i-fromIndex, j-fromIndex) = jacobian(i, j);
			}
		}
		permutation_matrix<double> p(space.width-fromIndex);
		lu_factorize(A, p);
		lu_substitute(A, p, b);//b holds the solution (change in given row of T)
		
		//put back in t-row
		for (unsigned int i = fromIndex; i < space.width; i++) {
			space.tMatrix(row, i) += b(i-fromIndex);
		}
	}
}
