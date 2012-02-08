#include "log.h"

void printVector(double * vector, int length) {
	for (int i = 0; i < length; i++) {
		std::cout << vector[i] << " ";
	}
	std::cout << "\n";
}
void printVector(double * vector, int length, string title) {
	cout << "\n" << title << " ";
	printVector(vector, length);
}

void printMatrix(double ** matrix, int height, int width) {
	for (int i = 0; i < height; i++) {
		printVector(matrix[i], width);
	}
}
void printMatrix(double ** matrix, int height, int width, string title) {
	cout << "\n" << title << "\n";
	printMatrix(matrix, height, width);
}