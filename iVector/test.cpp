#include "test.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>


using namespace boost::numeric::ublas;

/*
These methods are for testing the program. This includes both that the iVector/t-row updates are performed correctly, and estimations on the runtime
*/

#ifdef _WIN32
const static std::string TEST_BASEDIR_IN = "C:\\src\\cpp\\iVector\\testin\\";
const static std::string TEST_FILELIST_IN = "C:\\src\\cpp\\iVector\\testin\\filelist.txt";
const static std::string TEST_OUT_LOC = "C:\\src\\cpp\\iVector\\testout\\results.txt";
#else
const static std::string TEST_BASEDIR_IN = "./test/";
const static std::string TEST_FILELIST_IN = "./test/fileList.txt";
const static std::string TEST_OUT_LOC = "./test/results.txt";
#endif

const static double MICROS_IN_S = 1000000;


//Validity tests of the iVector program using a preset set of documents
void iVectTests() {
	
	std::vector<Document> documents;
	fetchDocumentsFromFileList(documents, TEST_FILELIST_IN, TEST_BASEDIR_IN, 3, 0, 1, 0, 2);
	std::cout << "\n\n--iVector tests, " << documents.size() << " docs found--\n\nFeature 4 for each document:";
	for (unsigned int i = 0; i<documents.size(); i++) {
		std::cout << " " << documents[i].getGammaValue(4);
	} 
	std::cout << "\nFeature 1:";
	for (unsigned int i = 0; i<documents.size(); i++) {
		std::cout << " " << documents[i].getGammaValue(1);
	} 
	
	FeatureSpace space(5, 3, documents, 23);
	printMatrix(space.tMatrix, "5x3 Random tMatrix:");
	
	matrix<double> tMatrix(5, 3);
	tMatrix(0,0) = 1;
	tMatrix(0,1) = 2;
	tMatrix(0,2) = 1;
	tMatrix(1,0) = 1;
	tMatrix(1,1) = 0;
	tMatrix(1,2) = 1;
	tMatrix(2,0) = 0;
	tMatrix(2,1) = 1;
	tMatrix(2,2) = 1;
	tMatrix(3,0) = 1;
	tMatrix(3,1) = 0;
	tMatrix(3,2) = 0;
	tMatrix(4,0) = 2;
	tMatrix(4,1) = 1;
	tMatrix(4,2) = 0;

	space = FeatureSpace(tMatrix, documents);
	printMatrix(space.tMatrix, "5x3 codeset tMatrix:");
	printVector(space.mVector, "mVector:");

	updateiVectors(documents, space, 2);
	
	std::cout << "\nInitial iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		std::cout << i << ": ";
		printVector(documents[i].iVector, "");
	}
	
	updatetRows(documents, space);
	printMatrix(space.tMatrix, "tMatrix update:");
	updateiVectors(documents, space);
	std::cout << "\nSecond iVector update values:\n";
	for (unsigned int i = 0; i < documents.size(); i++) {
		std::cout << i << ": ";
		printVector(documents[i].iVector, "");
	}
	std::cout << "\nTotal likelihood1 (1) = "<<calcTotalLikelihood(documents, space);
	updatetRows(documents, space, 2);
	std::cout << "\nTotal likelihood (1.5) = "<<calcTotalLikelihood(documents, space);
	updateiVectors(documents, space, 2);
	std::cout << "\nTotal likelihood (2) = "<<calcTotalLikelihood(documents, space);

	writeDocuments(documents, TEST_OUT_LOC);
}

//A speedtest on actual data. This will mostly give an estimate of the total runtime.
void speedTests(Configuration config) {
	#ifndef _WIN32
	int height = config.height;
	unsigned int width = config.width;
	int threads = config.threads;
	unsigned int updateNum = 5;

	std::cout << "\n\n--Speedtest--\n\n";
	struct timeval startTime;
	struct timeval stopTime;
	double time;

	gettimeofday(&startTime, NULL);
	std::vector<Document> traindocs = fetchDocumentsFromFileList(TRAINSET, config);
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Fetch " << traindocs.size() << " iVectors, " << time << " seconds\n";
    
    int docSize = traindocs.size();
	
	gettimeofday(&startTime, NULL);
	FeatureSpace space(height, width, traindocs, 23);
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Setup " << height << "x" << width << " feature space, " << time << " seconds\n";
	
	gettimeofday(&startTime, NULL);
	for (unsigned int i = 0; i < updateNum; i++) {
		updateiVector(traindocs[i], space);
	}
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Update " << updateNum << " iVectors, " << time << "s, update whole set, " << time/updateNum*traindocs.size() << " s.\n";
    
    gettimeofday(&startTime, NULL);
    vector<double> denominators = calcAllPhiDenominators(space, traindocs);
    gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Calculate all phi denominators, " << time << "s\n";
    
    gettimeofday(&startTime, NULL);
	for (unsigned int i = 0; i < updateNum; i++) {
		updatetRow(traindocs, space, i, denominators);
	}
	gettimeofday(&stopTime, NULL);
	time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Update " << updateNum << " rows of t, " << time << "s, update whole set, " << time/updateNum*space.height << " s.\n";
    
    
    
    while (traindocs.size() > width) {
        traindocs.pop_back();
    }
    
    gettimeofday(&startTime, NULL);
    updateiVectors(traindocs, space, threads);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << "Update " << width << " iVectors with " << threads << " threads, " << time << "s, update whole set, " << time/width*docSize << " s.\n";
    
    gettimeofday(&startTime, NULL);
    calcTotalLikelihood(traindocs, space);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
    std::cout << "Calculated likelihood for " << width << " documents in " << time << "s, calc for whole set, " << time/width*docSize << "s.\n";
    
    
    gettimeofday(&startTime, NULL);
    updatetRows(traindocs, space, threads);
    gettimeofday(&stopTime, NULL);
    time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
    std::cout << "Updated all t-rows with " << width << "iVectors using " << threads << " threads in " << time << "s, approx for whole set, " << time/width*docSize;
	#endif
}


//method for starting all tests.
void testAll(Configuration config) {
	iVectTests();
	speedTests(config);

	std::string breaker;
	getline(std::cin, breaker);
}