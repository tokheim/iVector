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
double calcAvgEucledianDistance(vector<Document> & documents) {
	double distance = 0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		distance += eucledianDistance(documents[i].iVector, documents[i].oldiVector);
	}
	return distance/documents.size();
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
double calcUtteranceLikelihoodExcludeInf(Document & document, FeatureSpace & space) {
    double likelihood = 0.0;
    double denominator = calcPhiDenominator(space, document.iVector);
    HASH_I_D::iterator it;
    for (it = document.gamma.begin(); it != document.gamma.end(); ++it) {
        double phi = calcPhi(space, document.iVector, it->first, denominator);
        if (log(phi) > MINUS_INF) {
            likelihood += it->second * log(phi);
        }
    }
    return likelihood;
}

double calcTotalLikelihoodExcludeInf(vector<Document> & documents, FeatureSpace & space) {
	double totLikelihood = 0.0;
	for (unsigned int i = 0; i < documents.size(); i++) {
		totLikelihood += calcUtteranceLikelihoodExcludeInf(documents[i], space);
	}
	return totLikelihood;
}
//Calculates both b vector (b=gradient) and jacobian for iVectors at once
void setUpSystem(vector<double> & b, vector< vector<double> > & jacobian, Document & document, FeatureSpace & space, double denominator) {
	size_t width = space.width;
	double * tRow;
	double * jacRow;
    size_t height = space.height;
	for (size_t row = 0; row < height; row++) {
		double gammaVal = document.getGammaValue(row);
		if (space.mVector[row] == MINUS_INF && gammaVal == 0.0) {
			continue;
		}
		tRow = &space.tMatrix[row][0];

		double phiPart = calcPhi(space, document.iVector, row, denominator)*document.gammaSum;
		double gradweight = gammaVal - phiPart;
		double jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		for (size_t i = 0; i < width; i++) {
			b[i] += gradweight*tRow[i];
			phiPart = tRow[i]*jacweight;
			jacRow = &jacobian[i][0];
			for (size_t j = i; j < width; j++) {
				jacRow[j] += tRow[j]*phiPart;
			}
		}
	}
	for (size_t i = 1; i < width; i++) {
		for (size_t j = 0; j < i; j++) {
			jacobian[i][j] = jacobian[j][i];
		}
	}
}
//Calculates both b vector (b=jacobian*iVector-gradient) and jacobian for rows of t at once
void setUpSystem(vector<double> &b, vector< vector<double> > & jacobian, vector<Document> & documents, FeatureSpace & space, int row, vector<double> &denominators) {
	size_t width = space.width;
    size_t docSize = documents.size();
	double * iVector;
	double * jacRow;
    double gammaVal;
    double phiPart;
    double gradweight;
    double jacweight;
	for (size_t n = 0; n < docSize; n++) {
		iVector = &documents[n].iVector[0];

		gammaVal = documents[n].getGammaValue(row);
		phiPart = calcPhi(space, documents[n].iVector, row, denominators[n])*documents[n].gammaSum;
		gradweight = gammaVal - phiPart;
		jacweight = phiPart;
		if (gradweight > 0.0) {
			jacweight = gammaVal;
		}
		for (size_t i = 0; i < width; i++) {
			b[i] += gradweight*iVector[i];
			phiPart = iVector[i]*jacweight;
			jacRow = &jacobian[i][0];
			for (size_t j = i; j < width; j++) {
				jacRow[j] += iVector[j]*phiPart;
			}
		}
	}
	for (size_t i = 1; i < width; i++) {
		for (size_t j = 0; j < i; j++) {
			jacobian[i][j] = jacobian[j][i];
		}
	}
}
//Decomposes matrix a, used to solve linear systems (cormen section 28.3)
void lupDecompose(vector< vector<double> > & a) {
	size_t dim = a.size();
    double *ai;
    double *aik;
    double *ak;
	for (size_t k = 0; k < dim; k++) {
		ak = &a[k][0];
        for (size_t i = k+1; i< dim; i++) {
			ai = &a[i][0];
            aik = &a[i][k];
            *aik = *aik/ak[k];
			for (size_t j = k+1; j < dim; j++) {
				ai[j] -= *aik * ak[j];
			}
		}
	}
}
//Solves lu-decomposed linear systems
vector<double> lupSolve(vector< vector<double> > &LU, vector<double> & b) {
	unsigned int dim = b.size();
	vector<double> x(dim);
	for (unsigned int i = 0; i < dim; i++) {
		x[i] = b[i];
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
	setUpSystem(b, jacobian, document, space, denominator);
	lupDecompose(jacobian);
	document.iVector = lupSolve(jacobian, b);
	addVectors(document.iVector, document.oldiVector);
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
		setUpSystem(b, jacobian, documents, space, row, denominators);
		lupDecompose(jacobian);
		space.tMatrix[row] = lupSolve(jacobian, b);
		addVectors(space.tMatrix[row], space.oldtMatrix[row]);

	}
}
void updatetRows(vector<Document> & documents, FeatureSpace & space) {
	vector<double> denominators = calcAllPhiDenominators(space, documents);
	for (unsigned int row = 0; row < space.height; row++) {
		updatetRow(documents, space, row, denominators);
	}
}

void recursiveiVectorUpdateCheck(Document & document, FeatureSpace & space, double oldLikelihood, int attempts) {
	if (oldLikelihood > calcUtteranceLikelihoodExcludeInf(document, space) && attempts < MAX_REDUCE_STEPSIZE_ATTEMPTS) {
		avgVector(document.iVector, document.oldiVector);
		recursiveiVectorUpdateCheck(document, space, oldLikelihood, attempts+1);
	}
	else if (attempts >= MAX_REDUCE_STEPSIZE_ATTEMPTS) {//Could not find a better iVector, so use the old one
		document.iVector = document.oldiVector;
	}
	//Else: the set iVector is better than the old one, do nothing
}
//Ensure that the update step of an iVector doesn't cause the likelihood to decrease
void updateiVectorCheckLike(Document & document, FeatureSpace & space) {
	double oldLikelihood = calcUtteranceLikelihoodExcludeInf(document, space);//could be changed from T-matrix updates
	updateiVector(document, space);
	recursiveiVectorUpdateCheck(document, space, oldLikelihood, 0);
}

void updateiVectorCheckLike(vector<Document> & documents, FeatureSpace & space) {
	for (unsigned int i = 0; i < documents.size(); i++) {
		updateiVectorCheckLike(documents[i], space);
	}
}
