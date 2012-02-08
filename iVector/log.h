#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <string>
using namespace std;

void printVector(double * vector, int length);
void printVector(double * vector, int length, string title);
void printMatrix(double ** matrix, int height, int width);
void printMatrix(double ** matrix, int height, int width, string title);

#endif