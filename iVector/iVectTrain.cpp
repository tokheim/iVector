#include "iVectTrain.h"

const int MAX_STEPS = 4;

void trainiVectors(string inFileList, string baseDir, string outLoc, int height, int width, unsigned int seed, bool limitFeature) {
	//Set up tMatrix and iVectors
	vector<Document> traindocs = fetchDocumentsFromFileList(TRAINSET, inFileList, baseDir, width, limitFeature);
	vector<Document> devtestdocs = fetchDocumentsFromFileList(DEVSET, inFileList, baseDir, width, limitFeature);
	FeatureSpace space(height, width, traindocs, seed);

	//Initially update all iVectors
	updateiVectors(traindocs, space);
	updateiVectors(devtestdocs, space);

	double oldLikelihood = log(0.0);
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	int steps = 0;

	while (newLikelihood > oldLikelihood && steps++ < MAX_STEPS) {
		oldLikelihood = newLikelihood;
		updatetRows(traindocs, space);
		updateiVectors(traindocs, space);
		updateiVectors(devtestdocs, space);
		newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	}
	space.tMatrix = space.oldtMatrix;
	useOldiVectors(traindocs);
	useOldiVectors(devtestdocs);

}

