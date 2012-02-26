#include "iVectTrain.h"

#include <sstream>

using namespace std;

const int MAX_STEPS = 9;

string intToString(int n) {
	stringstream ss;
	ss << n;
	return ss.str();
}



void trainiVectors(string inFileList, string baseDir, string outLoc, int height, int width, unsigned int seed, bool limitFeature, int threads) {
	resetClock();
	//Set up tMatrix and iVectors
	vector<Document> traindocs = fetchDocumentsFromFileList(TRAINSET, inFileList, baseDir, width, limitFeature);
	printTimeMsg("Done fetch train docs");
	vector<Document> devtestdocs = fetchDocumentsFromFileList(DEVSET, inFileList, baseDir, width, limitFeature);
	printTimeMsg("Done fetch devtest docs");
	FeatureSpace space(height, width, traindocs, seed);
	printTimeMsg("Done space setup");
	//Initially update all iVectors
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Done traindocs init");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Done devtest docs init");

	double oldLikelihood = log(0.0);
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	int steps = 0;
	bool doneSave = false;
	FeatureSpace optimalSpace = space;
    
    cout << "\nDevtest like ei (0): " << newLikelihood << "\nTrain like (0): " << calcTotalLikelihood(traindocs, space) << "\n";
	//while (newLikelihood > oldLikelihood && steps++ < MAX_STEPS) {
    while (steps++ < MAX_STEPS) {
        cout << "\n---Start step " << steps << "---\n";
		printTimeMsg("time?");
		oldLikelihood = newLikelihood;
		updatetRows(traindocs, space, threads);
		printTimeMsg("Updated t rows");
		updateiVectors(traindocs, space, threads);
		printTimeMsg("Updated train iVectors");
		updateiVectors(devtestdocs, space, threads);
		printTimeMsg("Updated devtest iVectors");
		newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
		printTimeMsg("Calc likelihood");
        cout << "Devtest like ei (" << steps << "): " << newLikelihood << "\n";
		cout << "Train like (" << steps << "): " << calcTotalLikelihood(traindocs, space) << "\n";
		
		writeDocuments(traindocs, outLoc+string("train")+intToString(steps));
		writeDocuments(devtestdocs, outLoc+string("devtest")+intToString(steps));
		printTimeMsg("Writing done");
		if (newLikelihood < oldLikelihood && !doneSave) {
			optimalSpace.tMatrix = space.oldtMatrix;
			doneSave = true;
		}
	}
	
	printTimeMsg("Finished loop");
	//space.tMatrix = space.oldtMatrix;
	//useOldiVectors(traindocs);
	//useOldiVectors(devtestdocs);
    
    /*string outPath = outLoc+"train10.txt";
    writeDocuments(traindocs, outPath);
    outPath = outLoc+"devtest10.txt";
    writeDocuments(traindocs, outPath);
	printTimeMsg("Completly finished");*/

	resetiVectors(traindocs);
	resetiVectors(devtestdocs);
	vector<Document> testdocs = fetchDocumentsFromFileList(EVLSET, inFileList, baseDir, width, limitFeature);
	
	cout << "Sizes; train: " << traindocs.size() << " devtest: " << devtestdocs.size() << " test: " << testdocs.size() << "\n";

	steps = 0;
	printTimeMsg("Ready for reset loop\n\n\n");

	while (steps++ < MAX_STEPS) {
		cout << "\n--Start step " << steps << "--\n";
		updateiVectors(traindocs, optimalSpace, threads);
		printTimeMsg("Train updated");
		updateiVectors(devtestdocs, optimalSpace, threads);
		printTimeMsg("Devtest updated");
		updateiVectors(testdocs, optimalSpace, threads);
		printTimeMsg("EVLtest updated");

		writeDocuments(traindocs, outLoc+string("Rtrain")+intToString(steps));
		writeDocuments(devtestdocs, outLoc+string("Rdevtest")+intToString(steps));
		writeDocuments(testdocs, outLoc+string("Revltest")+intToString(steps));
		printTimeMsg("docs saved");

		cout << "Train likelihood: " << calcTotalLikelihoodExcludeInf(traindocs, optimalSpace) << "\n";
		cout << "Devtest likelihood: " << calcTotalLikelihoodExcludeInf(devtestdocs, optimalSpace) << "\n";
		cout << "Test likelihood: " << calcTotalLikelihoodExcludeInf(testdocs, optimalSpace) << "\n";
		printTimeMsg("Likelihoods finished calculated");
		
		double euclDist = 0.0;
		for (size_t j = 0; j < traindocs.size(); j++) {
			euclDist += eucledianDistance(traindocs[j].iVector, traindocs[j].oldiVector);
		}
		euclDist = euclDist/traindocs.size();
		cout << "Train distance = " << euclDist << "\n";
		
		euclDist = 0.0;
		for (size_t j = 0; j < testdocs.size(); j++) {
			euclDist += eucledianDistance(testdocs[j].iVector, testdocs[j].oldiVector);
		}
		euclDist = euclDist/testdocs.size();
		cout << "Test distance = " << euclDist << "\n";

		euclDist = 0.0;
		for (size_t j = 0; j < devtestdocs.size(); j++) {
			euclDist += eucledianDistance(devtestdocs[j].iVector, devtestdocs[j].oldiVector);
		}
		euclDist = euclDist/devtestdocs.size();
		cout << "Devtest distance = " << euclDist << "\n";
		printTimeMsg("Eucledian distances calculated");
	}
	printTimeMsg("All done\n");
}

