#include "log.h"
#include <boost/numeric/ublas/io.hpp>

#ifndef _WIN32
#include <sys/time.h>
struct timeval startTime;
#endif

using namespace boost::numeric::ublas;

const static double MICROS_IN_S = 1000000;

void resetClock() {
	#ifndef _WIN32
	gettimeofday(&startTime, NULL);
	#endif
}
void printMsg(std::string msg) {
	std::cout << msg << "\n";
	std::cout.flush();
}
void printTimeMsg(std::string msg) {
	#ifndef _WIN32
	struct timeval stopTime;
	gettimeofday(&stopTime, NULL);
	double time = stopTime.tv_sec-startTime.tv_sec+((stopTime.tv_usec-startTime.tv_usec)/MICROS_IN_S);
	std::cout << time << " - ";
	#endif
	std::cout << msg << "\n";
	std::cout.flush();
}


void printVector(vector<double> &vect, std::string title) {
	std::cout << "\n" << title << " " << vect << "\n";
}

void printMatrix(matrix<double> &matrix, std::string title) {
	std::cout << "\n" << title << "\n" << matrix << "\n";
}