#include "iVectIO.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/algorithm/string.hpp>

const static double MINUS_INF = log(0.0);

using namespace std;

/*
Theese methods performs the programs IO operations, like reading spoken document vectors and saving iVectors
*/


typedef pair <string, int> S_I_PAIR;

//A mapping of string names to numbers
const static S_I_PAIR LANGUAGES[] = {S_I_PAIR("ENG_GENRL", 1), S_I_PAIR("SPANISH", 2), S_I_PAIR("MANDARIN_M", 3), S_I_PAIR("FARSI", 4),
	S_I_PAIR("FRENCH_CAN", 5), S_I_PAIR("GERMAN", 6), S_I_PAIR("HINDI", 7), S_I_PAIR("JAPANESE", 8), S_I_PAIR("KOREAN", 9), S_I_PAIR("TAMIL", 10),
	S_I_PAIR("VIETNAMESE", 11), S_I_PAIR("ARABIC_EGYPT", 12), S_I_PAIR("ENG_SOUTH", 14), S_I_PAIR("SPANISH_CAR", 15), S_I_PAIR("MANDARIN_T", 16),
	S_I_PAIR("ENGLISH", 1), S_I_PAIR("MANDARIN", 3), S_I_PAIR("FRENCH", 5), S_I_PAIR("ARABIC", 12)};
const static int OUT_OF_SET_LANG_NUM = 13;//All languages not found in the above list will be mapped to this language
const static int LANG_LIST_LENGTH = 19;

const static string SET_INPUTFILE_NAMES[] = {"train_list.txt", "devtest_list.txt", "nist_list.txt", "short_train_list.txt"};

int getLanguageClass(string & language) {
	for (int i = 0; i < LANG_LIST_LENGTH; i++) {
		if (LANGUAGES[i].first == language) {
			return LANGUAGES[i].second;
		}
	}
	return OUT_OF_SET_LANG_NUM;
}

//Reads a single document
Document readDocument(int languageClass, string & fullpath, int dim, int featureNameCol, int featureValueCol) {
	ifstream inFile;
	inFile.open(fullpath.c_str());
	if (!inFile.is_open()) {
		cerr << "Unable to read file " << fullpath;
		exit(1);
	}
	
	HASH_I_D featureMap;
	string line;

	while (getline(inFile, line)) {
		vector<string> splitLine;
		boost::split(splitLine, line, boost::is_any_of("\t "));
		int feature = atoi(splitLine[featureNameCol].c_str());
		double value = atof(splitLine[featureValueCol].c_str());
		featureMap.insert(I_D_PAIR(feature, value));
	}
	inFile.close();
	Document doc(languageClass, featureMap, dim);
	return doc;
}

//Reads all the languages from a given file list and appends it to the given list of documents
void fetchDocumentsFromFileList(vector<Document> & documents, string fullPath, string baseDir, int dim, int languageCol, int fileNameCol, int featureNameCol, int featureValueCol) {
	ifstream inFile;
	inFile.open(fullPath.c_str());
	if (!inFile.is_open()) {
		cerr << "Unable to read file " << fullPath;
		exit(1);
	}
	string line;
	while (getline(inFile, line)) {
		vector<string> splitLine;
		boost::split(splitLine, line, boost::is_any_of("\t "));
		int languageClass = getLanguageClass(splitLine[languageCol]);
		string filePath = baseDir+splitLine[fileNameCol];
		documents.push_back(readDocument(languageClass, filePath, dim, featureNameCol, featureValueCol));
	}
	inFile.close();
}

//Reads the documents given in the file list (directory found from config, actual file given by the speechSet value)
vector<Document> fetchDocumentsFromFileList(int speechSet, Configuration &config) {
	vector<Document> documents;
	fetchDocumentsFromFileList(documents, config.fileListInDir+SET_INPUTFILE_NAMES[speechSet], config.baseDir, config.width, 0, 1, 0, config.featureColumn);	
	return documents;
}
//Saves the tMatrix and mVector to the given directory.
void writeSpace(FeatureSpace & space, string fullPath) {
	ofstream outFile;
	outFile.open(fullPath.c_str());
	if (!outFile.is_open()) {
		cerr << "Unable to write to file " << fullPath;
		exit(1);
	}
	outFile << space.mVector(0);
	for (unsigned int i = 1; i < space.height; i++) {
		if (space.mVector(i) != MINUS_INF) {
			outFile << " " << space.mVector(i);
		} else {
			outFile << " -inf";
		}
	}
	
	for (unsigned int i = 0; i < space.height; i++) {
		outFile << "\n" << space.tMatrix(i, 0);
		for (unsigned int j = 1; j < space.width; j++) {
			outFile << " " << space.tMatrix(i, j);
		}
	}
	outFile.close();
}
//Reads the mVector and tMatrix from the given directory
FeatureSpace readSpace(Configuration & config) {
	ifstream inFile;
	inFile.open(config.featureSpacePath.c_str());
	if (!inFile.is_open()) {
		cerr << "Unable to open file " << config.featureSpacePath;
		exit(1);
	}
	string line;
	getline(inFile, line);
	vector<string> splitLine;
	boost::split(splitLine, line, boost::is_any_of("\t "));
	boost::numeric::ublas::vector<double> mVector(config.height);
	for (int i = 0; i < config.height; i++) {
		mVector(i) = atof(splitLine[i].c_str());
		if (splitLine[i].find("-inf") != string::npos) {
			mVector(i) = MINUS_INF;
		}
	}
	boost::numeric::ublas::matrix<double> tMatrix(config.height, config.width);
	int row = 0;
	while (getline(inFile, line)) {
		boost::split(splitLine, line, boost::is_any_of("\t "));
		for (unsigned int i = 0; i < splitLine.size(); i++) {
			tMatrix(row, i) = atof(splitLine[i].c_str());
		}
		row += 1;
	}
	inFile.close();
	return FeatureSpace(tMatrix, mVector);
}

//Appends an iVector to a filestream (in liblinear format)
void writeDocument(ofstream & outFile, Document & document) {
	outFile << document.languageClass;
	for (unsigned int i = 0; i < document.iVector.size(); i++) {
		outFile << " " << (i+1) << ":" << document.iVector(i);
	}
	outFile << "\n";
}

//Writes a list of documents
void writeDocuments(vector<Document> & documents, string fullPath) {
	ofstream outFile;
	outFile.open(fullPath.c_str());
	if (!outFile.is_open()) {
		cerr << "Unable to write to file " << fullPath;
		exit(1);
	}
	for (unsigned int i = 0; i < documents.size(); i++) {
		writeDocument(outFile, documents[i]);
	}
	outFile.close();
}