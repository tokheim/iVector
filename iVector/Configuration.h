#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#include <string>


struct Configuration {
	int height;
	int width;
	int seed;
	int featureColumn;
	int threads;
	std::string outLoc;
	std::string baseDir;
	std::string fileListInDir;
	std::string featureSpacePath;
	bool loadFeatureSpace;
	bool useTwoTrainSets;
	
	Configuration(int argc, char *argv[]);
	std::string toString();
};
#endif