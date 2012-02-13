#include "iVectTrain.h"
using namespace std;

const int MAX_STEPS = 10;

void trainiVectors(string inFileList, string baseDir, string outLoc, int height, int width, unsigned int seed, bool limitFeature, int threads) {
	//Set up tMatrix and iVectors
	vector<Document> traindocs = fetchDocumentsFromFileList(TRAINSET, inFileList, baseDir, width, limitFeature);
	vector<Document> devtestdocs = fetchDocumentsFromFileList(DEVSET, inFileList, baseDir, width, limitFeature);
	FeatureSpace space(height, width, traindocs, seed);

	//Initially update all iVectors
	updateiVectors(traindocs, space, threads);
	updateiVectors(devtestdocs, space, threads);

	double oldLikelihood = log(0.0);
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	int steps = 0;
    
    cout << "Devtest likelihood excluding inf (0): " << newLikelihood << "\nTrain like (0): " << calcTotalLikelihood(traindocs, space) << "\n";
	//while (newLikelihood > oldLikelihood && steps++ < MAX_STEPS) {
    while (steps++ < MAX_STEPS) {
        oldLikelihood = newLikelihood;
		updatetRows(traindocs, space, threads);
		updateiVectors(traindocs, space, threads);
		updateiVectors(devtestdocs, space, threads);
		newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
        cout << "Devtest likelihood excluding inf (" << steps << "): " << newLikelihood << "\nTrain like (" << steps << "): " << calcTotalLikelihood(traindocs, space) << "\n";
	}
	space.tMatrix = space.oldtMatrix;
	useOldiVectors(traindocs);
	useOldiVectors(devtestdocs);
    
    string outPath = outLoc+"train.txt";
    writeDocuments(traindocs, outPath);
    outPath = outLoc+"devtest.txt";
    writeDocuments(traindocs, outPath);

}

