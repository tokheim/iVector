#include "log.h"


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