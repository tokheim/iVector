#include "configuration.h"
#include <sstream>
#include <math.h>
#include <iostream>
using namespace std;

/*
This class  holds all the different options for iVector extraction, parsing input arguments, and displaying usage
*/

const static string PARAM_IN_FILELIST_DIR = "-i";
const static string PARAM_IN_BASE_DIR = "-b";
const static string PARAM_OUT_LOC = "-o";
const static string PARAM_TRIGRAM_COUNT = "-C";
const static string PARAM_IVECT_COUNT = "-r";
const static string PARAM_SEED = "-S";
const static string PARAM_FEATURE_COLUMN = "-L";
const static string PARAM_THREADS = "-t";
const static string PARAM_LOADSPACE = "-l";
const static string PARAM_USE_TWO_TRAIN_SETS = "-T";

const static string HELP_TEXT = "Lorem ipsum dolor est C Full map 205379, mapped with bi and unigram 37059, unmapped with bi and unigram 208919";

void printCommandError(string cmd);
void printParamError(string cmd, char * paramValue);
void setPositiveValue(string paramName, char * sValue, int *param);
bool parseBool(string sValue);

Configuration::Configuration(int argc, char *argv[]) {
	//Fill default values
	fileListInDir = "./other/";
	baseDir = "";
	outLoc = "./iVectors/";
	threads = 8;
	featureColumn = 1;
	seed = 23;
	width = 200;
	height = 35937;//Full map 205379, mapped with bi and unigram 37059, unmapped with bi and unigram 208919
	featureSpacePath = "";
	loadFeatureSpace = false;
	useTwoTrainSets = false;
	
	//Parse values from argc/argv
	for (int i = 1; i < argc-1; i+=2) {
		string paramName = string(argv[i]);
		if (paramName == PARAM_IN_FILELIST_DIR) {
			fileListInDir = argv[i+1];
		}
		else if (paramName == PARAM_IN_BASE_DIR) {
			baseDir = argv[i+1];
		}
		else if (paramName == PARAM_OUT_LOC) {
			outLoc = argv[i+1];
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
		else if (paramName == PARAM_LOADSPACE) {
			featureSpacePath = argv[i+1];
			loadFeatureSpace = true;
		}
		else if (paramName == "?" || paramName == "-?") {
			cout << HELP_TEXT;
			exit(0);
		}
		else if (paramName == PARAM_FEATURE_COLUMN) {
			setPositiveValue(paramName, argv[i+1], featureColumn);
		}
		else if (paramName == PARAM_USE_TWO_TRAIN_SETS) {
			useTwoTrainSets = parseBool(string(argv[i+1]));
		}
		else {
			printCommandError(paramName);
		}
	}
}

string Configuration::toString() {
	stringstream ss;
	ss << "File list directory: " << fileListInDir << "\n";
	ss << "Base directory for documents: " << baseDir << "\n";
	ss << "Save directory: " << outLoc << "\n";
	ss << "Threads available: " << threads << "\n";
	ss << "Column to read feature counts: " << featureColumn << "\n";
	ss << "Seed for T-matrix generation: " << seed << "\n";
	ss << "iVector size: " << width << "\n";
	ss << "height: " << height << "\n";
	if (loadFeatureSpace) {
		ss << "Load feature space from " << featureSpacePath << "\n";
	} 
	ss << "Use different sets for T and classifier training " << useTwoTrainSets << "\n";
	return ss.str();
}

bool parseBool(string sValue) {
	return sValue=="true" || sValue=="1" || sValue=="True" || sValue=="TRUE";
}

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