#include "log.h"

#ifndef _WIN32
#include <sys/time.h>
struct timeval startTime;
#endif

const static double MICROS_IN_S = 1000000;

void resetClock() {
	#ifndef _WIN32
	gettimeofday(&startTime, NULL);
	#endif
}
void printMsg(string msg) {
	cout << msg << "\n";
	cout.flush();
}
void printTimeMsg(string msg) {
	#ifndef _WIN32
	struct timeval stopTime;
	gettimeofday(&stopTime, NULL);
	double time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	cout << time << " - ";
	#endif
	cout << msg << "\n";
	cout.flush();
}



void printVector(vector<double> &vect) {
	for (unsigned int i = 0; i < vect.size(); i++) {
		std::cout << vect[i] << " ";
	}
	std::cout << "\n";
}
void printVector(vector<double> &vect, string title) {
	cout << "\n" << title << " ";
	printVector(vect);
}

void printMatrix(vector< vector<double> > &matrix) {
	for (unsigned int i = 0; i < matrix.size(); i++) {
		printVector(matrix[i]);
	}
}
void printMatrix(vector< vector<double> > &matrix, string title) {
	cout << "\n" << title << "\n";
	printMatrix(matrix);
}