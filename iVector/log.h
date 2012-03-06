#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <string>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

void printVector(boost::numeric::ublas::vector<double> &vect, std::string title);
void printMatrix(boost::numeric::ublas::matrix<double> &matrix, std::string title);
void resetClock();
void printTimeMsg(std::string msg);
void printMsg(std::string msg);

#endif