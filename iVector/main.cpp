#include <iostream>
#include "test.h"
#include "iVectTrain.h"
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

const static string DEF_IN_FILELIST_DIR = "./other/";
const static string DEF_IN_BASE_DIR = "";
const static string DEF_OUT_LOC = "./iVectors/";
const static int DEF_TRIGRAM_COUNT = 35937;
const static int DEF_IVECT_DIM = 50;
const static int DEF_SEED = 23;
const static int DEF_THREADS = 8;
const static bool DEF_LIMIT_FEATURE = true;

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
	string inFileListDir = DEF_IN_FILELIST_DIR;
	string outLocation = DEF_OUT_LOC;
	string inBaseDir = DEF_IN_BASE_DIR;
	int height = DEF_TRIGRAM_COUNT;
	int width = DEF_IVECT_DIM;
	int seed = DEF_SEED;
    int threads = DEF_THREADS;
	bool limitFeature = DEF_LIMIT_FEATURE;
	
	for (int i = 1; i < argc-1; i+=2) {
		string paramName = string(argv[i]);
		if (paramName == PARAM_IN_FILELIST_DIR) {
			inFileListDir = argv[i+1];
		}
		else if (paramName == PARAM_IN_BASE_DIR) {
			inBaseDir = argv[i+1];
		}
		else if (paramName == PARAM_OUT_LOC) {
			outLocation = argv[i+1];
		}
		else if (paramName == PARAM_TRIGRAM_COUNT) {
			setPositiveValue(paramName, argv[i+1], &height);
		}
		else if (paramName == PARAM_IVECT_COUNT) {
			setPositiveValue(paramName, argv[i+1], &width);
		}
		else if (paramName == PARAM_SEED) {
			setPositiveValue(paramName, argv[i+1], &seed);
		}
        else if (paramName == PARAM_THREADS) {
            setPositiveValue(paramName, argv[i+1], &threads);
        }
		else if (paramName == "?" || paramName == "-?") {
			cout << HELP_TEXT;
			return 0;
		}
		else if (paramName == PARAM_LIMIT_FEATURE) {
			string paramValue = string(argv[i+1]);
            limitFeature = paramValue=="true" || paramValue=="1" || paramValue=="True";
		}
		else {
			printCommandError(paramName);
		}
	}
	cout << "in " << inFileListDir << " base " << inBaseDir << " out " << outLocation << " h " << height << " w " << width << " s " << seed << "\n";

	
	testAll(width, threads);
    //trainiVectors(inFileListDir, inBaseDir, outLocation, height, width, seed, limitFeature, threads);
    
    
	return 0;
}
