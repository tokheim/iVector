#include "iVectMath.h"

const static int MAX_REDUCE_STEPSIZE_ATTEMPTS = 4;
const static double MINUS_INF = log(0.0);


//Allocates a vector with all zero values
double * initializeVector(int length) {
	double * vector = new double[length];
	for (int i = 0; i < length; i++) {
		vector[i] = 0.0;
	}
	return vector;
}

//Allocates a matrix with all zero values
double ** initializeMatrix(int height, int width) {
	double ** matrix = new double * [height];
	for (int i = 0; i < height; i++) {
		matrix[i] = initializeVector(width);
	}
	return matrix;
}

bool isZeroVector(double * vector, int length) {
	for (int i = 0; i < length; i++) {
		if (vector[i] != 0.0 && vector[i] != -0.0) {
			return false;
		}
	}
	return true;
}

double * addVectors(double * vectora, double * vectorb, int length) {
	double * newvector = new double[length];
	for (int i = 0; i < length; i++) {
		newvector[i] = vectora[i]+vectorb[i];
	}
	return newvector;
}
double multiplyVectors(double * rowvector, double * columnvector, int length) {
	double sum = 0;
	for (int i = 0; i < length; i++) {
		sum += rowvector[i]*columnvector[i];
	}
	return sum;
}
double * scaleVector(double * vector, double scalar, int length) {
	double * newvector = new double[length];
	for (int i = 0; i < length; i++) {
		newvector[i] = vector[i]*scalar;
	}
	return newvector;
}
//Scales the vector "vector", adds the result to "addToVector"
void scaleAndAddVector(double * addToVector, double * vector, double scalar, int length) {
	for (int i = 0; i < length; i++) {
		addToVector[i] += vector[i]*scalar;
	}
}

double eucledianDistance(double * vectora, double * vectorb, int length) {
	double distance = 0;
	for (int i = 0; i < length; i++) {
		distance += pow(vectora[i]-vectorb[i], 2);
	}
	return distance;
}
//usefull if the newton rapshon update steps are too big
double * avgVector(double * vectorA, double * vectorB, int dim) {
	return scaleVector(addVectors(vectorA, vectorB, dim), 0.5, dim);
}

//Calculates the denominator in phi for a given iVector
double calcPhiDenominator(FeatureSpace & space, double * iVector) {
	double denominator = 0.0;
	for (int i = 0; i < space.height; i++) {
		denominator += exp(space.mVector[i]+multiplyVectors(space.tMatrix[i], iVector, space.width));
	}
	return denominator;
}
//Above but for more iVectors
double * calcAllPhiDenominators(FeatureSpace & space, vector<Document> & documents) {
	double * denominators = new double[documents.size()];
	for (unsigned int i = 0; i < documents.size(); i++) {
		denominators[i] = calcPhiDenominator(space, documents[i].iVector);
	}
	return denominators;
}
double calcPhi(FeatureSpace & space, double * iVector, int row, double denominator) {
	return exp(space.mVector[row]+multiplyVectors(space.tMatrix[row], iVector, space.width))/denominator;
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

double calcTotalLikelihood(vector<Document> & documents, FeatureSpace & space) {
	double likelihood = 0.0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		likelihood += calcUtteranceLikelihood(documents[i], space);
	}
	return likelihood;
}
double calcTotalLikelihoodExcludeInf(vector<Document> & documents, FeatureSpace & space) {
	double totLikelihood = 0.0;
	double likelihood;
	for (unsigned int i = 0; i < documents.size(); i++) {
		likelihood = calcUtteranceLikelihood(documents[i], space);
		if (likelihood > MINUS_INF) {
			totLikelihood += likelihood;
		}
	}
	return totLikelihood;
}
//Calculates both b vector (b=jacobian*ivector-gradient) and jacobian for iVectors at once
void setUpSystem(double * b, double ** jacobian, Document & document, FeatureSpace & space, double denominator) {
	int docIndex = 0;
	for (int row = 0; row < space.height; row++) {
		double gradweight;
		double jacweight;
		if (document.gammaList[docIndex].first == row) {// gamma for row exists
			double phiPart = calcPhi(space, document.iVector, row, denominator)*document.gammaSum;
			gradweight = document.gammaList[docIndex].second - phiPart;
			jacweight = phiPart;
			if (document.gammaList[docIndex].second > phiPart) {
				jacweight = document.gammaList[docIndex].second;
			}
			if (docIndex < document.uniqueGammas -1) {
				docIndex++;
			}
		}
		else if (space.mVector[row] != MINUS_INF) {// gamma for row is zero
			jacweight = calcPhi(space, document.iVector, row, denominator)*document.gammaSum;
			gradweight = -jacweight;
		}
		else {
			continue;
		}
		for (int i = 0; i < space.width; i++) {
			b[i] -= gradweight*space.tMatrix[row][i];
			for (int j = 0; j < space.width; j++) {
				jacobian[i][j] -= space.tMatrix[row][i]*space.tMatrix[row][j]*jacweight;
			}
		}
	}
	for (int i = 0; i < space.width; i++) {
		for (int j = 0; j < space.width; j++) {
			b[i] += jacobian[i][j]*document.iVector[j];
		}
	}



	/*for (int row = 0; row < space.height; row++) {
		double gammaVal = document.getGammaValue(row);
		if (space.mVector[row] == MINUS_INF && gammaVal == 0.0) {
			continue;
		}
		double phiPart = calcPhi(space, document.iVector, row, denominator)*document.gammaSum;
		double gradweight = gammaVal - phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		for (int i = 0; i < space.width; i++) {
			b[i] -= gradweight*space.tMatrix[row][i];
			for (int j = 0; j < space.width; j++) {
				jacobian[i][j] -= space.tMatrix[row][i]*space.tMatrix[row][j]*jacweight;
			}
		}
	}
	for (int i = 0; i < space.width; i++) {
		for (int j = 0; j < space.width; j++) {
			b[i] += jacobian[i][j]*document.iVector[j];
		}
	}*/
}
//Calculates both b vector (b=jacobian*iVector-gradient) and jacobian for rows of t at once
void setUpSystem(double *b, double ** jacobian, vector<Document> & documents, FeatureSpace & space, int row, double * denominators) {
	for (unsigned int n = 0; n < documents.size(); n++) {
		double gammaVal = documents[n].getGammaValue(row);
		double phiPart = calcPhi(space, documents[n].iVector, row, denominators[n])*documents[n].gammaSum;
		double gradweight = gammaVal - phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		for (int i = 0; i < space.width; i++) {
			b[i] -= gradweight*documents[n].iVector[i];
			for (int j = 0; j < space.width; j++) {
				jacobian[i][j] -= documents[n].iVector[i]*documents[n].iVector[j]*jacweight;
			}
		}
	}
	for (int i = 0; i < space.width; i++) {
		for (int j = 0; j < space.width; j++) {
			b[i] += jacobian[i][j]*space.tMatrix[row][j];
		}
	}
}



//Calculates b in linear system Ax=b from x=oldVector - jacobian^-1*gradient
double * calcBVector(double * oldVector, double *gradient, double ** jacobian, int dim) {
	double * b = new double[dim];
	for (int i = 0; i < dim; i++) {
		b[i] = -gradient[i];
		for (int j = 0; j < dim; j++) {
			b[i] += jacobian[i][j] * oldVector[j];
		}
	}
	return b;
}
//Decomposes matrix a, returns permutation p, used to solve linear systems (cormen section 28.3)
int * lupDecompose(double ** a, int dim) {
	int * p = new int[dim];
	for (int i = 0; i < dim; i++) {
		p[i] = i;
	}
	for (int k = 0; k < dim; k++) {
		double max = 0.0;
		int newrow = -1;
		for (int i = k; i < dim; i++) {
			if (fabs(a[i][k]) > max) {
				max = fabs(a[i][k]);
				newrow = i;
			}
		}
		if (newrow == -1) {
			std::cerr << "\n\nERROR: Method lupDecompose failed, jacobian hasn't full rank";
			//exit(1);
			newrow = k;
		}

		int temprowvalue = p[k];
		p[k] = p[newrow];
		p[newrow] = temprowvalue;
		for (int i = 0; i < dim; i++) {
			double temp = a[k][i];
			a[k][i] = a[newrow][i];
			a[newrow][i] = temp;
		}
		for (int i = k+1; i< dim; i++) {
			a[i][k] = a[i][k]/a[k][k];
			for (int j = k+1; j < dim; j++) {
				a[i][j] = a[i][j]-a[i][k]*a[k][j];
			}
		}
	}
	return p;
}
//Solves lup-decomposed linear systems
double * lupSolve(double ** LU, double * b, int * p, int dim) {
	double * x = new double[dim];
	for (int i = 0; i < dim; i++) {
		x[i] = b[p[i]];
		for (int j = 0; j < i; j++) {
			x[i] -= LU[i][j] * x[j];
		}
	}
	for (int i = dim -1; i >= 0; i--) {
		for (int j = i+1; j < dim; j++) {
			x[i] -= LU[i][j] * x[j];
		}
		x[i] = x[i]/LU[i][i];
	}
	return x;
}
//Calculate one newton rapshon step
double * calcNewtonStep(double * oldVector, double * gradient, double ** jacobian, int dim) {
	double * b = calcBVector(oldVector, gradient, jacobian, dim);
	int * p = lupDecompose(jacobian, dim);
	return lupSolve(jacobian, b, p, dim);
}

//Calculates the gradient of the given row of matrix t
double * calcTGradient(vector<Document> & documents, FeatureSpace & space, double * denominators, int row) {
	double * gradient = initializeVector(space.width);
	for (unsigned int n = 0; n < documents.size(); n++) {
		double weight = documents[n].getGammaValue(row) - documents[n].gammaSum * calcPhi(space, documents[n].iVector, row, denominators[n]);
		scaleAndAddVector(gradient, documents[n].iVector, weight, space.width);
	}
	return gradient;
}
//Calculates an iVectors gradient
double * calciVectGradient(Document & document, FeatureSpace & space, double denominator) {
	double * gradient = initializeVector(space.width);
	for (int row = 0; row < space.height; row++) {
		double gammaVal = document.getGammaValue(row);
		if (space.mVector[row] != MINUS_INF || gammaVal != 0.0) {
			double weight = gammaVal - document.gammaSum * calcPhi(space, document.iVector, row, denominator);
			scaleAndAddVector(gradient, space.tMatrix[row], weight, space.width);
		}
	}
	return gradient;
}
//Does part of the calculation for the jacobian for both rows of t and iVectors
void calcJacobianSumStep(double ** jacobian, Document & document, FeatureSpace & space, int row, double denominator, double * vector) {
	double weight = std::max(document.getGammaValue(row), document.gammaSum * calcPhi(space, document.iVector, row, denominator));
	for (int i = 0; i < space.width; i++) {
		for (int j = 0; j < space.width; j++) {
			jacobian[i][j] -= vector[i]*vector[j]*weight;
		}
	}
}

//Returns the approximated Jacobian for a row from matrix t
double ** approxtRowJacobian(vector<Document> & documents, FeatureSpace & space, double * denominators, int row) {
	double ** jacobian = initializeMatrix(space.width, space.width);
	for (unsigned int n = 0; n < documents.size(); n++) {
		calcJacobianSumStep(jacobian, documents[n], space, row, denominators[n], documents[n].iVector);
	}
	return jacobian;
}
//Returns the approximated jacobian for an iVector
double ** approxiVectorJacobian(Document & document, FeatureSpace & space,  double denominator) {
	double ** jacobian = initializeMatrix(space.width, space.width);
	for (int row = 0; row < space.height; row++) {
		if (space.mVector[row] != MINUS_INF || document.getGammaValue(row) != 0.0) {
			calcJacobianSumStep(jacobian, document, space, row, denominator, space.tMatrix[row]);
		}
	}
	return jacobian;
}

void updateiVector(Document & document, FeatureSpace & space) {
	document.oldiVector = document.iVector;
	double denominator = calcPhiDenominator(space, document.iVector);
	
	
	//new
	double * b = initializeVector(space.width);
	double ** jacobian = initializeMatrix(space.width, space.width);
	setUpSystem(b, jacobian, document, space, denominator);
	int * p = lupDecompose(jacobian, space.width);
	document.iVector = lupSolve(jacobian, b, p, space.width);
	
	
	/*
	//old
	double * gradient = calciVectGradient(document, space, denominator);
	if (!isZeroVector(gradient, space.width)) {//If gradient is zero then update is unneccessary/mathematically illegal
		double ** jacobian = approxiVectorJacobian(document, space, denominator);
		document.iVector = calcNewtonStep(document.iVector, gradient, jacobian, space.width);
	}
	else {
		std::cout << "Warning, iVector gradient is zero";
	}
	*/
}
void updateiVectors(vector<Document> & documents, FeatureSpace & space) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		updateiVector(documents[i], space);
	}
}
void updatetRow(vector<Document> & documents, FeatureSpace & space, int row, double * denominators) {
	space.oldtMatrix[row] = space.tMatrix[row];
	if (space.mVector[row] != MINUS_INF) {//If mVector is trained on a superset of "documents" then the old values should still be a solution
		double * b = initializeVector(space.width);
		double ** jacobian = initializeMatrix(space.width, space.width);
		setUpSystem(b, jacobian, documents, space, row, denominators);
		int * p = lupDecompose(jacobian, space.width);
		space.tMatrix[row] = lupSolve(jacobian, b, p, space.width);

		/*
		double * gradient = calcTGradient(documents, space, denominators, row);
		double ** jacobian = approxtRowJacobian(documents, space, denominators, row);
		space.tMatrix[row] = calcNewtonStep(space.tMatrix[row], gradient, jacobian, space.width);
		*/
	}
}
void updatetRows(vector<Document> & documents, FeatureSpace & space) {
	double * denominators = calcAllPhiDenominators(space, documents);
	for (int row = 0; row < space.height; row++) {
		updatetRow(documents, space, row, denominators);
	}
}




bool recursiveiVectorUpdateCheck(Document & document, FeatureSpace & space, double oldLikelihood, int attempts) {
	if (attempts <= MAX_REDUCE_STEPSIZE_ATTEMPTS) {
		if (oldLikelihood > calcUtteranceLikelihood(document, space)) {
			document.iVector = avgVector(document.iVector, document.oldiVector, space.width);
			return recursiveiVectorUpdateCheck(document, space, oldLikelihood, attempts+1);
		}
		return true;
	}
	return false;
}

void updateiVectorsCheckStep(vector<Document> & documents, FeatureSpace & space) {
	for (unsigned int n = 0; n < documents.size(); n++) {
		double oldLikelihood = calcUtteranceLikelihood(documents[n], space);
		updateiVector(documents[n], space);
		recursiveiVectorUpdateCheck(documents[n], space, oldLikelihood, 0);
	}
}
bool recursivetRowUpdateCheck(vector<Document> & documents, FeatureSpace & space, double oldTotLikelihood, int attempts) {
	if (attempts <= MAX_REDUCE_STEPSIZE_ATTEMPTS) {
		if (oldTotLikelihood > calcTotalLikelihood(documents, space)) {
			for (int row = 0; row < space.height; row++) {
				space.tMatrix[row] = avgVector(space.tMatrix[row], space.oldtMatrix[row], space.width);
			}
			return recursivetRowUpdateCheck(documents, space, oldTotLikelihood, attempts+1);
		}
		return true;
	}
	return false;
}

/*
Since it might be possible that the update steps are too great, this updater will constraint the
update step if the total likelihood after update has decreased. False is returned if it is unable
to increase likelihood
*/
bool updatetRowsCheckStep(vector<Document> & documents, FeatureSpace & space) {
	double oldTotLikelihood = calcTotalLikelihood(documents, space);
	updatetRows(documents, space);
	return recursivetRowUpdateCheck(documents, space, oldTotLikelihood, 0);
}