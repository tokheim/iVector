#include <iostream>
#include "test.h"
#include "iVectTrain.h"
#include "configuration.h"
using namespace std;

/*
Main method loading configurations and starting the program
*/

int main(int argc, char *argv[]) {
	Configuration config(argc, argv);
	cout << config.toString();
	testAll(config);
	//trainiVectors(config);
	return 0;
}
