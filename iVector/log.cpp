#include "log.h"
#include <boost/numeric/ublas/io.hpp>

#ifndef _WIN32
#include <sys/time.h>
struct timeval startTime;
#endif

/*
Class for logging information
*/

using namespace boost::numeric::ublas;

const static double MICROS_IN_S = 1000000;

//This method resets the timer used to timestamp log transcriptions
void resetClock() {
	#ifndef _WIN32
	gettimeofday(&startTime, NULL);
	#endif
}

void printMsg(std::string msg) {
	std::cout << msg << "\n";
	std::cout.flush(); //Performs flushing so that events are immediatly visible when pipeing logs to files
}
void printTimeMsg(std::string msg) {
	#ifndef _WIN32
	struct timeval stopTime;
	gettimeofday(&stopTime, NULL);
	double time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << time << " - ";
	#endif
	printMsg(msg);
}


void printVector(vector<double> &vect, std::string title) {
	std::cout << "\n" << title << " " << vect << "\n";
}

void printMatrix(matrix<double> &matrix, std::string title) {
	std::cout << "\n" << title << "\n" << matrix << "\n";
}