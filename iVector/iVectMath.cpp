#include "iVectMath.h"

const static int MAX_REDUCE_STEPSIZE_ATTEMPTS = 4;
const static double MINUS_INF = log(0.0);



void addVectors(vector<double> &addToVector, vector<double> &addFromVector) {
	for (unsigned int i = 0; i < addToVector.size(); i++) {
		addToVector[i] += addFromVector[i];
	}
}

double multiplyVectors(vector<double> &rowvector, vector<double> &columnvector) {
	double sum = 0;
	for (unsigned int i = 0; i < rowvector.size(); i++) {
		sum += rowvector[i]*columnvector[i];
	}
	return sum;
}

//Scales the vector "vector", adds the result to "addToVector"
void scaleAndAddVector(vector<double> &addToVector, vector<double> &scalevector, double scalar) {
	for (unsigned int i = 0; i < addToVector.size(); i++) {
		addToVector[i] += scalevector[i]*scalar;
	}
}

double eucledianDistance(vector<double> &vectora, vector<double> &vectorb) {
	double distance = 0;
	for (unsigned int i = 0; i < vectora.size(); i++) {
		distance += pow(vectora[i]-vectorb[i], 2);
	}
	return distance;
}
//usefull if the newton rapshon update steps are too big
void avgVector(vector<double> &saveToVector, vector<double> &unchangedVector) {
	for (unsigned int i = 0; i < saveToVector.size(); i++) {
		saveToVector[i] = (saveToVector[i]+unchangedVector[i])/2;
	}
}

//Calculates the denominator in phi for a given iVector
double calcPhiDenominator(FeatureSpace & space, vector<double> &iVector) {
	double denominator = 0.0;
	for (unsigned int i = 0; i < space.height; i++) {
		denominator += exp(space.mVector[i]+multiplyVectors(space.tMatrix[i], iVector));
	}
	return denominator;
}
//Above but for more iVectors
vector<double> calcAllPhiDenominators(FeatureSpace & space, vector<Document> & documents) {
	vector<double> denominators(documents.size());
	for (unsigned int i = 0; i < documents.size(); i++) {
		denominators[i] = calcPhiDenominator(space, documents[i].iVector);
	}
	return denominators;
}
double calcPhi(FeatureSpace & space, vector<double> &iVector, int row, double denominator) {
	return exp(space.mVector[row]+multiplyVectors(space.tMatrix[row], iVector))/denominator;
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
void setUpSystem(vector<double> & b, vector< vector<double> > & jacobian, Document & document, FeatureSpace & space, double denominator) {
	for (unsigned int row = 0; row < space.height; row++) {
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
		for (unsigned int i = 0; i < space.width; i++) {
			b[i] -= gradweight*space.tMatrix[row][i];
			for (unsigned int j = i; j < space.width; j++) {
				jacobian[i][j] -= space.tMatrix[row][i]*space.tMatrix[row][j]*jacweight;
			}
		}
	}
	for (unsigned int i = 0; i < space.width; i++) {
		for (unsigned int j = 0; j < i; j++) {
			jacobian[i][j] = jacobian[j][i];
		}
		for (unsigned int j = 0; j < space.width; j++) {
			b[i] += jacobian[i][j]*document.iVector[j];
		}
	}
}
//Calculates both b vector (b=jacobian*iVector-gradient) and jacobian for rows of t at once
void setUpSystem(vector<double> &b, vector< vector<double> > & jacobian, vector<Document> & documents, FeatureSpace & space, int row, vector<double> denominators) {
	for (unsigned int n = 0; n < documents.size(); n++) {
		double gammaVal = documents[n].getGammaValue(row);
		double phiPart = calcPhi(space, documents[n].iVector, row, denominators[n])*documents[n].gammaSum;
		double gradweight = gammaVal - phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		for (unsigned int i = 0; i < space.width; i++) {
			b[i] -= gradweight*documents[n].iVector[i];
			for (unsigned int j = i; j < space.width; j++) {
				jacobian[i][j] -= documents[n].iVector[i]*documents[n].iVector[j]*jacweight;
			}
		}
	}
	for (unsigned int i = 0; i < space.width; i++) {
		for (unsigned int j = 0; j < i; j++) {
			jacobian[i][j] = jacobian[j][i];
		}
		for (unsigned int j = 0; j < space.width; j++) {
			b[i] += jacobian[i][j]*space.tMatrix[row][j];
		}
	}
}
//Decomposes matrix a, returns permutation p, used to solve linear systems (cormen section 28.3)
void lupDecompose(vector< vector<double> > & a, vector<int> & p) {
	unsigned int dim = p.size();
	for (unsigned int i = 0; i < dim; i++) {
		p[i] = i;
	}
	for (unsigned int k = 0; k < dim; k++) {
		double max = 0.0;
		int newrow = -1;
		for (unsigned int i = k; i < dim; i++) {
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
		//SHOULD MOVE WHOLE ROW
		for (unsigned int i = 0; i < dim; i++) {
			double temp = a[k][i];
			a[k][i] = a[newrow][i];
			a[newrow][i] = temp;
		}
		for (unsigned int i = k+1; i< dim; i++) {
			a[i][k] = a[i][k]/a[k][k];
			for (unsigned int j = k+1; j < dim; j++) {
				a[i][j] = a[i][j]-a[i][k]*a[k][j];
			}
		}
	}
}
//Solves lup-decomposed linear systems
vector<double> lupSolve(vector< vector<double> > &LU, vector<double> & b, vector<int> p) {
	unsigned int dim = p.size();
	vector<double> x(dim);
	for (unsigned int i = 0; i < dim; i++) {
		x[i] = b[p[i]];
		for (unsigned int j = 0; j < i; j++) {
			x[i] -= LU[i][j] * x[j];
		}
	}
	for (unsigned int i = dim -1; i < dim; i--) {//Underflows at negative..
		for (unsigned int j = i+1; j < dim; j++) {
			x[i] -= LU[i][j] * x[j];
		}
		x[i] = x[i]/LU[i][i];
	}
	return x;
}

void updateiVector(Document & document, FeatureSpace & space) {
	document.oldiVector = document.iVector;
	double denominator = calcPhiDenominator(space, document.iVector);
	vector<double> b(space.width, 0.0);
	vector< vector<double> > jacobian(space.width, vector<double>(space.width, 0.0));
	vector<int> p(space.width);
	setUpSystem(b, jacobian, document, space, denominator);
	lupDecompose(jacobian, p);
	document.iVector = lupSolve(jacobian, b, p);
}
void updateiVectors(vector<Document> & documents, FeatureSpace & space) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		updateiVector(documents[i], space);
	}
}
void updatetRow(vector<Document> & documents, FeatureSpace & space, int row, vector<double> & denominators) {
	space.oldtMatrix[row] = space.tMatrix[row];
	if (space.mVector[row] != MINUS_INF) {//If mVector is trained on a superset of "documents" then the old values should still be a solution
		vector<double> b(space.width, 0.0);
		vector< vector<double> > jacobian(space.width, vector<double>(space.width, 0.0));
		vector<int> p(space.width);
		setUpSystem(b, jacobian, documents, space, row, denominators);
		lupDecompose(jacobian, p);
		space.tMatrix[row] = lupSolve(jacobian, b, p);

	}
}
void updatetRows(vector<Document> & documents, FeatureSpace & space) {
	vector<double> denominators = calcAllPhiDenominators(space, documents);
	for (unsigned int row = 0; row < space.height; row++) {
		updatetRow(documents, space, row, denominators);
	}
}





/*
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
/*
bool updatetRowsCheckStep(vector<Document> & documents, FeatureSpace & space) {
	double oldTotLikelihood = calcTotalLikelihood(documents, space);
	updatetRows(documents, space);
	return recursivetRowUpdateCheck(documents, space, oldTotLikelihood, 0);
}
*/