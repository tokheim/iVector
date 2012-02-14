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
	/*
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
	*/
}
void matrixTests() {
	/*
	Ax = b should give x-values [-1.4, 2.2, 0.6]
	*/
	cout << "\n\n--matrix tests-- \n\n";
	vector< vector<double> > matrix(3, vector<double>(3));
	matrix[0][0] = 1;
	matrix[0][1] = 2;
	matrix[0][2] = 0;
	matrix[1][0] = 3;
	matrix[1][1] = 4;
	matrix[1][2] = 4;
	matrix[2][0] = 5;
	matrix[2][1] = 6;
	matrix[2][2] = 3;

	vector<double> bvector(3);
	bvector[0] = 3;
	bvector[1] = 7;
	bvector[2] = 8;
	
	//vector<int> p(3);

	printMatrix(matrix, "A:");
	printVector(bvector, "b-vector: ");
	lupDecompose(matrix);
	vector<double> x = lupSolve(matrix, bvector);
	printVector(x, "Ax = b -> x = ");
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
	/*cout << "\nFeaturelists:\n";
	for (unsigned int i = 0; i<documents.size(); i++) {
		for (int j = 0; j < documents[i].uniqueGammas; j++) {
			cout << documents[i].gammaList[j].first << " ";
		}
		cout << "\n";
	}*/
	
	FeatureSpace space(5, 3, documents, 23);
	printMatrix(space.tMatrix, "5x3 Random tMatrix:");
	
	vector< vector<double> > tMatrix(5, vector<double>(3));
	tMatrix[0][0] = 1;
	tMatrix[0][1] = 2;
	tMatrix[0][2] = 1;
	tMatrix[1][0] = 1;
	tMatrix[1][1] = 0;
	tMatrix[1][2] = 1;
	tMatrix[2][0] = 0;
	tMatrix[2][1] = 1;
	tMatrix[2][2] = 1;
	tMatrix[3][0] = 1;
	tMatrix[3][1] = 0;
	tMatrix[3][2] = 0;
	tMatrix[4][0] = 2;
	tMatrix[4][1] = 1;
	tMatrix[4][2] = 0;

	space = FeatureSpace(tMatrix, documents);
	printMatrix(space.tMatrix, "5x3 codeset tMatrix:");
	printVector(space.mVector, "mVector:");

	updateiVectors(documents, space, 2);
	
	cout << "\nInitial iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		cout << i << ": ";
		printVector(documents[i].iVector);
	}
	
	updatetRows(documents, space, 2);
	printMatrix(space.tMatrix, "tMatrix update:");
	updateiVectors(documents, space);
	cout << "\nSecond iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		cout << i << ": ";
		printVector(documents[i].iVector);
	}
	cout << "\nTotal likelihood1 (1) = "<<calcTotalLikelihood(documents, space);
	updatetRows(documents, space);
	cout << "\nTotal likelihood (1.5) = "<<calcTotalLikelihood(documents, space);
	updateiVectors(documents, space);
	cout << "\nTotal likelihood (2) = "<<calcTotalLikelihood(documents, space);

	writeDocuments(documents, TEST_OUT_LOC);
}

//System dependent
void speedTests(int width, int threads) {
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
    
    int docSize = traindocs.size();
	
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
    
    gettimeofday(&startTime, NULL);
    vector<double> denominators = calcAllPhiDenominators(space, traindocs);
    gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Calculate all phi denominators, " << time << "s\n";
    
    gettimeofday(&startTime, NULL);
	for (int i = 0; i < updateNum; i++) {
		updatetRow(traindocs, space, i, denominators);
	}
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Update " << updateNum << " rows of t, " << time << "s, update whole set, " << time/updateNum*space.height << " s.\n";
    
    
    
    while (traindocs.size() > width) {
        traindocs.pop_back();
    }
    
    gettimeofday(&startTime, NULL);
    updateiVectors(traindocs, space, threads);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << "Update " << width << " iVectors with " << threads << " threads, " << time << "s, update whole set, " << time/width*docSize << " s.\n";
    
    gettimeofday(&startTime, NULL);
    calcTotalLikelihood(traindocs, space);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
    cout << "Calculated likelihood for " << width << " documents in " << time << "s, calc for whole set, " << time/width*docSize << "s.\n";
    
    
    gettimeofday(&startTime, NULL);
    updatetRows(traindocs, space, threads);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
    cout << "Updated all t-rows with " << width << "iVectors using " << threads << " threads in " << time << "s, approx for whole set, " << time/width*docSize;
	#endif
}



void testAll(int width, int threads) {
	vectorTests();
	matrixTests();
	iVectTests();
	speedTests(width, threads);

	string breaker;
	getline(cin, breaker);
}