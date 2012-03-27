#include "iVectTrain.h"

#include <sstream>

/*
High level methods implementing the procedures for finding the best T-matrix, and extracting the iVectors from this matrix
*/

using namespace std;

void extractiVectors(vector<Document> traindocs, vector<Document> devtestdocs, vector<Document> testdocs, FeatureSpace & space, int threads, string outLoc);
void branchTraining(vector<Document> & traindocs, vector<Document> & devtestdocs, vector<Document> & testdocs, FeatureSpace & space, string outLoc, int threads);
void traintMatrix(vector<Document> & traindocs, vector<Document> & devtestdocs, FeatureSpace & space, string outLoc, int threads);

const int MAX_TRAIN_STEPS = 7;
const int MAX_EXTRACT_STEPS = 9;




string intToString(int n) {
	stringstream ss;
	ss << n;
	return ss.str();
}

string doubleToString(double num) {
	stringstream ss;
	ss << num;
	return ss.str();
}

//The method for training t, and extract iVectors from this matrix
void trainiVectors(Configuration config) {
	resetClock();
	//Set up tMatrix and iVectors
	vector<Document> traindocs;
	if (config.useTwoTrainSets && !config.loadFeatureSpace) {
		traindocs = fetchDocumentsFromFileList(STRAINSET, config);
	}
	else if (!config.loadFeatureSpace) {
		traindocs = fetchDocumentsFromFileList(TRAINSET, config);
	}
	printTimeMsg(string("Fetched ")+intToString(traindocs.size())+string(" train docs"));
	vector<Document> devtestdocs = fetchDocumentsFromFileList(DEVSET, config);
	printTimeMsg(string("Fetched ")+intToString(devtestdocs.size())+string(" devtest docs"));
	
	FeatureSpace space(config.height, config.width, traindocs, config.seed);
	if (!config.loadFeatureSpace) {
		printTimeMsg("Done space setup");

		traintMatrix(traindocs, devtestdocs, space, config.outLoc, config.threads);

		writeSpace(space, config.outLoc+string("space"));
	}
	else {
		space = readSpace(config);
		printTimeMsg("Loaded space");
	}

	vector<Document> testdocs = fetchDocumentsFromFileList(EVLSET, config);
	printTimeMsg(string("Fetched ")+intToString(testdocs.size())+string(" evltest docs"));
	
	if (config.useTwoTrainSets) {
		traindocs = fetchDocumentsFromFileList(TRAINSET, config);
		printTimeMsg(string("Fetched ")+intToString(traindocs.size())+string(" train docs"));
	}


	//branchTraining(traindocs, devtestdocs, testdocs, space, outLoc, threads);

	extractiVectors(traindocs, devtestdocs, testdocs, space, config.threads, config.outLoc);
}
//Reset iVectors for each update iteration (Recursive check for likelihoods is not really neccessary)
double doResetUpdateIteration(vector<Document> & traindocs, vector<Document> & devtestdocs, FeatureSpace & space, int threads) {
	resetiVectors(traindocs);
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Updated train iVectors");
	resetiVectors(devtestdocs);
	updatetRows(traindocs, space, threads);
	printTimeMsg("Updated t rows");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Updated devtest iVectors");
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	
	//Nice to print but strictly unneccessary
	printTimeMsg(string("Train avg likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(traindocs, space)/traindocs.size()));
	printTimeMsg(string("Devtest avg likelihood ")+doubleToString(newLikelihood/devtestdocs.size()));
	
	printTimeMsg(string("Train avg distance ")+doubleToString(calcAvgEuclideanDistance(traindocs)));
	printTimeMsg(string("Devtest avg distance ")+doubleToString(calcAvgEuclideanDistance(devtestdocs)));
	

	return newLikelihood;
}

//Update iteratation of both iVectors and t-matrix
double doUpdateIteration(vector<Document> & traindocs, vector<Document> & devtestdocs, FeatureSpace & space, int threads) {
	updatetRows(traindocs, space, threads);
	printTimeMsg("Updated t rows");
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Updated train iVectors");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Updated devtest iVectors");
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	
	//Nice to print but strictly unneccessary
	printTimeMsg(string("Train avg likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(traindocs, space)/traindocs.size()));
	printTimeMsg(string("Devtest avg likelihood ")+doubleToString(newLikelihood/devtestdocs.size()));
	
	printTimeMsg(string("Train avg distance ")+doubleToString(calcAvgEuclideanDistance(traindocs)));
	printTimeMsg(string("Devtest avg distance ")+doubleToString(calcAvgEuclideanDistance(devtestdocs)));
	

	return newLikelihood;
}
//Will extract iVectors for each update iteration of t
void branchTraining(vector<Document> & traindocs, vector<Document> & devtestdocs, vector<Document> & testdocs, FeatureSpace & space, string outLoc, int threads) {
	printTimeMsg("---Start initial branch---\n");
	extractiVectors(traindocs, devtestdocs, testdocs, space, threads, outLoc+string("it0"));
	
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Done traindocs init");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Done devtest docs init");
	int steps = 0;

	while (steps++ < MAX_TRAIN_STEPS) {
		printTimeMsg(string("---Start step ") + intToString(steps) + string("---\n"));
		doUpdateIteration(traindocs, devtestdocs, space, threads);
		printTimeMsg("---Start branch---\n");

		extractiVectors(traindocs, devtestdocs, testdocs, space, threads, outLoc+string("it")+intToString(steps));
	}
}



//Finds the t-Matrix giving highest likelihood to devtest documents
void traintMatrix(vector<Document> & traindocs, vector<Document> & devtestdocs, FeatureSpace & space, string outLoc, int threads) {
	//Initially update all iVectors
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Done traindocs init");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Done devtest docs init");

	double oldLikelihood = log(0.0);
	double newLikelihood = calcTotalLikelihoodExcludeInf(devtestdocs, space);
	int steps = 0;
    
    printTimeMsg(string("Train likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(traindocs, space)/traindocs.size()));
	printTimeMsg(string("Devtest likelihood ")+doubleToString(newLikelihood/devtestdocs.size()));
	
	//while (newLikelihood > oldLikelihood && steps++ < MAX_STEPS) {
    while (steps++ < MAX_TRAIN_STEPS) {
		printTimeMsg(string("---Start step ") + intToString(steps) + string("---\n"));
		oldLikelihood = newLikelihood;
		
		newLikelihood = doUpdateIteration(traindocs, devtestdocs, space, threads);

		writeDocuments(traindocs, outLoc+string("train")+intToString(steps));
		writeDocuments(devtestdocs, outLoc+string("devtest")+intToString(steps));
		printTimeMsg("Writing done");
		if (newLikelihood < oldLikelihood) {
			printTimeMsg("Overtrained, using previous t-Matrix;");
			space.tMatrix = space.oldtMatrix;
			break;
		}
	}
	printTimeMsg("Finished t-training");
}

//Do one update iteration on iVectors from train, dev and evaluation sets, and print results.
void doUpdateIteration(vector<Document> & traindocs, vector<Document> & devtestdocs, vector<Document> & testdocs, FeatureSpace & space, int threads) {
	updateiVectors(traindocs, space, threads);
	printTimeMsg("Train updated");
	updateiVectors(devtestdocs, space, threads);
	printTimeMsg("Devtest updated");
	updateiVectors(testdocs, space, threads);
	printTimeMsg("Evltest updated");

	//Strictly unneccessary but nice to see
	printTimeMsg(string("Train likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(traindocs, space)/traindocs.size()));
	printTimeMsg(string("Devtest likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(devtestdocs, space)/devtestdocs.size()));
	printTimeMsg(string("Evltest likelihood ")+doubleToString(calcTotalLikelihoodExcludeInf(testdocs, space)/testdocs.size()));
	
	printTimeMsg(string("Train distance") + doubleToString(calcAvgEuclideanDistance(traindocs)));
	printTimeMsg(string("Devtest distance") + doubleToString(calcAvgEuclideanDistance(devtestdocs)));
	printTimeMsg(string("Test distance") + doubleToString(calcAvgEuclideanDistance(testdocs)));
}

//Method for extracting iVectors from a given T-matrix. Saves iVectors after each iteration. Currently copies list of documents for usage with branching
void extractiVectors(vector<Document> traindocs, vector<Document> devtestdocs, vector<Document> testdocs, FeatureSpace & space, int threads, string outLoc) {
	
	resetiVectors(traindocs);
	resetiVectors(devtestdocs);
	resetiVectors(testdocs);//Shouldn't be neccessary except for branch-testing

	int steps = 0;
	printTimeMsg("Ready for reset loop\n\n\n");

	while (steps++ < MAX_EXTRACT_STEPS) {
		printTimeMsg(string("--Start step ") + intToString(steps) + string("--\n"));

		doUpdateIteration(traindocs, devtestdocs, testdocs, space, threads);

		writeDocuments(traindocs, outLoc+string("Rtrain")+intToString(steps));
		writeDocuments(devtestdocs, outLoc+string("Rdevtest")+intToString(steps));
		writeDocuments(testdocs, outLoc+string("Revltest")+intToString(steps));
		printTimeMsg("docs saved");
	}
	printTimeMsg("All done\n");
}