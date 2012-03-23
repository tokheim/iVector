#include <iostream>
#include "test.h"
#include "iVectTrain.h"
#include "configuration.h"
#include <string>
#include <math.h>
using namespace std;

const static string PARAM_IN_FILELIST_DIR = "-i";
const static string PARAM_IN_BASE_DIR = "-b";
const static string PARAM_OUT_LOC = "-o";
const static string PARAM_TRIGRAM_COUNT = "-C";
const static string PARAM_IVECT_COUNT = "-r";
const static string PARAM_SEED = "-s";
const static string PARAM_LIMIT_FEATURE = "-L";
const static string PARAM_THREADS = "-t";

const static string HELP_TEXT = "Lorem ipsum dolor est";

void printCommandError(string cmd) {
	cout << "Parameter \"" << cmd << "\" not known.\nUsage:\n" << HELP_TEXT;
	exit(1);
}
void printParamError(string cmd, char * paramValue) {
	cout << "Value \"" << paramValue << "\" is illegal for parameter \"" << cmd << "\".\nUsage:\n" << HELP_TEXT;
	exit(1);
}
void setPositiveValue(string paramName, char * sValue, int *param) {
	*param = atoi(sValue);
	if (*param <= 0) {
		printParamError(paramName, sValue);
	}
}

int main(int argc, char *argv[]) {
	Configuration config;
	
	for (int i = 1; i < argc-1; i+=2) {
		string paramName = string(argv[i]);
		if (paramName == PARAM_IN_FILELIST_DIR) {
			config.fileListInDir = argv[i+1];
		}
		else if (paramName == PARAM_IN_BASE_DIR) {
			config.baseDir = argv[i+1];
		}
		else if (paramName == PARAM_OUT_LOC) {
			config.outLoc = argv[i+1];
		}
		else if (paramName == PARAM_TRIGRAM_COUNT) {
			setPositiveValue(paramName, argv[i+1], &config.height);
		}
		else if (paramName == PARAM_IVECT_COUNT) {
			setPositiveValue(paramName, argv[i+1], &config.width);
		}
		else if (paramName == PARAM_SEED) {
			setPositiveValue(paramName, argv[i+1], &config.seed);
		}
        else if (paramName == PARAM_THREADS) {
            setPositiveValue(paramName, argv[i+1], &config.threads);
        }
		else if (paramName == "?" || paramName == "-?") {
			cout << HELP_TEXT;
			return 0;
		}
		else if (paramName == PARAM_LIMIT_FEATURE) {
			string paramValue = string(argv[i+1]);
            config.limitFeatures = paramValue=="true" || paramValue=="1" || paramValue=="True";
		}
		else {
			printCommandError(paramName);
		}
	}
	cout << config.toString();
	
	testAll(config);
	//trainiVectors(config);
	//shortTrainiVectors(config);
    
	return 0;
}
