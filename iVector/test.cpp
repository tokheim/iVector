#include "test.h"

#ifdef _WIN32
const static string TEST_BASEDIR_IN = "C:\\src\\cpp\\iVector\\testin\\";
const static string TEST_FILELIST_IN = "C:\\src\\cpp\\iVector\\testin\\filelist.txt";
const static string TEST_OUT_LOC = "C:\\src\\cpp\\iVector\\testout\\results.txt";
#else
const static string TEST_BASEDIR_IN = "./test/";
const static string TEST_FILELIST_IN = "./test/fileList.txt";
const static string TEST_OUT_LOC = "./test/results.txt";
const static string TEST_SPEED_FILELIST_LOC = "./other/";
const static string TEST_SPEED_BASEDIR = "";
#endif

const static double MICROS_IN_S = 1000000;

void vectorTests() {
	
	cout << "--vector tests--\n\n";
	double a[] = {2.0, 5.0, 1.0};
	printVector(a, 3, "a = ");
	double b[] = {1.0, 3.0, 2.0};
	printVector(b, 3, "b = ");
	printVector(addVectors(a, b, 3), 3, "a+b = ");
	cout << "a*b^T = " << multiplyVectors(a, b, 3) << "\n";
	printVector(scaleVector(a, 5.0, 3), 3, "a*5 = ");
	cout << "eucledianDist = " << eucledianDistance(a, b, 3) << "\n";
	printVector(avgVector(a, b, 3), 3, "(a+b)/2 = ");
	scaleAndAddVector(b, a, 5.0, 3);
	printVector(b, 3, "a*5+b = ");
}
void matrixTests() {
	/*
	Ax = b should give x-values [-1.4, 2.2, 0.6]
	*/
	cout << "\n\n--matrix tests-- \n\n";
	double ** matrix = new double * [3];
	double row0[] = {1, 2, 0};
	double row1[] = {3, 4, 4};
	double row2[] = {5, 6, 3};
	matrix[0] = row0;
	matrix[1] = row1;
	matrix[2] = row2;
	
	printMatrix(initializeMatrix(2, 2), 2, 2, "Initialized 2x2 matrix");

	double bvector[] = {3, 7, 8};
	printMatrix(matrix, 3, 3, "A:");
	printVector(bvector, 3, "b-vector: ");
	int * p = lupDecompose(matrix, 3);
	printVector(lupSolve(matrix, bvector, p, 3), 3, "Ax = b -> x = ");
}



void iVectTests() {
	/* 
	With 3 documents with "gamma"-values (set number of languages in iVectIO to 1)
	gamma0 = [7.0, 2.0, 0.0, 0.0, 3.0];
	gamma1 = [3.0, 0.0, 1.0, 0.0, 3.0];
	gamma2 = [4.0, 1.0, 5.0, 0.0, 2.0];
	The total likelihood (2) should end up being -33.4543
	With L languages, total likelihood (2) should be -33.4543*L
	*/
	
	//vector<Document> documents = fetchDocuments(0, TEST_IN_LOC, 3);
	vector<Document> documents;
	fetchDocumentsFromFileList(documents, TEST_FILELIST_IN, TEST_BASEDIR_IN, 3, 0, 1, 0, 2);
	cout << "\n\n--iVector tests, " << documents.size() << " docs found--\n\nFeature 4 for each document:";
	for (unsigned int i = 0; i<documents.size(); i++) {
		cout << " " << documents[i].getGammaValue(4);
	} 
	cout << "\nFeature 1:";
	for (unsigned int i = 0; i<documents.size(); i++) {
		cout << " " << documents[i].getGammaValue(1);
	} 
	FeatureSpace * space = new FeatureSpace(5, 3, documents, 23);
	printMatrix(space->tMatrix, 5, 3, "5x3 Random tMatrix:");
	
	double ** tMatrix = new double * [5];
	double row0[] = {1, 2, 1};
	double row1[] = {1, 0, 1};
	double row2[] = {0, 1, 1};
	double row3[] = {1, 0, 0};
	double row4[] = {2, 1, 0};
	tMatrix[0] = row0;
	tMatrix[1] = row1;
	tMatrix[2] = row2;
	tMatrix[3] = row3;
	tMatrix[4] = row4;

	space = new FeatureSpace(5, 3, tMatrix, documents);
	printMatrix(space->tMatrix, 5, 3, "5x3 codeset tMatrix:");
	printVector(space->mVector, 5, "mVector:");

	updateiVectors(documents, * space);
	
	cout << "\nInitial iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		cout << i << ": ";
		printVector(documents[i].iVector, 3);
	}
	
	updatetRows(documents, * space);
	printMatrix(space->tMatrix, 5, 3, "tMatrix update:");
	updateiVectors(documents, * space);
	cout << "\nSecond iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		cout << i << ": ";
		printVector(documents[i].iVector, 3);
	}
	cout << "\nTotal likelihood1 (1) = "<<calcTotalLikelihood(documents, * space);
	updatetRows(documents, * space);
	cout << "\nTotal likelihood (1.5) = "<<calcTotalLikelihood(documents, * space);
	updateiVectors(documents, * space);
	cout << "\nTotal likelihood (2) = "<<calcTotalLikelihood(documents, * space);

	writeDocuments(documents, TEST_OUT_LOC, 3);
}

//System dependent
void speedTests(int width) {
	#ifndef _WIN32
	int height = 50653;
	int updateNum = 5;

	cout << "\n\n--Speedtest--\n\n";
	struct timeval startTime;
	struct timeval stopTime;
	double time;

	gettimeofday(&startTime, NULL);
	vector<Document> traindocs = fetchDocumentsFromFileList(TRAINSET, TEST_SPEED_FILELIST_LOC, TEST_SPEED_BASEDIR, width, false);
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Fetch " << traindocs.size() << " iVectors, " << time << " seconds\n";
	
	gettimeofday(&startTime, NULL);
	FeatureSpace space(height, width, traindocs, 23);
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Setup " << height << "x" << width << " feature space, " << time << " seconds\n";
	
	gettimeofday(&startTime, NULL);
	for (int i = 0; i < updateNum; i++) {
		updateiVector(traindocs[i], space);
	}
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Update " << updateNum << " iVectors, " << time << "s, update whole set, " << time/updateNum*traindocs.size() << " s.\n";
	#endif
}



void testAll(int width) {
	vectorTests();
	matrixTests();
	iVectTests();
	speedTests(width);

	string breaker;
	getline(cin, breaker);
}