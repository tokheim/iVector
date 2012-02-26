#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

void printVector(vector<double> &vect);
void printVector(vector<double> &vect, string title);
void printMatrix(vector< vector<double> > &matrix);
void printMatrix(vector< vector<double> > &matrix, string title);
void resetClock();
void printTimeMsg(string msg);
void printMsg(string msg);

#endif