#include "iVectIO.h"

//typedef pair <int, double> I_D_PAIR;
typedef pair <string, int> S_I_PAIR;

//Language->Languageid
const static S_I_PAIR LANGUAGES[] = {S_I_PAIR("ARABIC_EGYPT", 1), S_I_PAIR("ENG_GENRL", 2), S_I_PAIR("ENG_SOUTH", 2), S_I_PAIR("FARSI", 3),
	S_I_PAIR("FRENCH_CAN", 4), S_I_PAIR("GERMAN", 5), S_I_PAIR("HINDI", 6), S_I_PAIR("JAPANESE", 7), S_I_PAIR("KOREAN", 8), S_I_PAIR("MANDARIN_M", 9), 
	S_I_PAIR("MANDARIN_T", 9), S_I_PAIR("SPANISH", 10), S_I_PAIR("SPANISH_CAR", 10), S_I_PAIR("TAMIL", 11), S_I_PAIR("VIETNAMESE", 12) };
const static int NUMBER_OF_LANGUAGES = 15;//full = 15
//const static string TRANS_EXTENSION = ".txt";
//const static string TRANS_SEARCH_PATTERN = "*"+TRANS_EXTENSION;
//const static string SPEECH_SETS[] = {"train_raw", "devtest_raw", "evltest_raw"};
const static string SET_INPUTFILE_NAMES[] = {"train_list.txt", "devtest_list.txt", "evltest_list.txt"};


//Very basic string splitting (no regards to double spaces and such)
vector<string> splitString(string & s, char splitsign) {
	vector<string> svect;
	size_t startpos = 0;
	size_t endpos = s.find(splitsign);
	while (endpos != string::npos) {
		svect.push_back(s.substr(startpos, endpos-startpos));
		startpos = endpos+1;
		endpos = s.find(splitsign, startpos);
	}
	svect.push_back(s.substr(startpos));
	return svect;
}
int getLanguageClass(string & language) {
	for (int i = 0; i < NUMBER_OF_LANGUAGES; i++) {
		if (LANGUAGES[i].first == language) {
			return LANGUAGES[i].second;
		}
	}
	return -1;
}

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
		vector<string> splitLine = splitString(line, ' ');
		int feature = atoi(splitLine[featureNameCol].c_str());
		double value = atof(splitLine[featureValueCol].c_str());
		featureMap.insert(I_D_PAIR(feature, value));
	}
	inFile.close();
	Document doc(languageClass, featureMap, dim);
	return doc;
}

void fetchDocumentsFromFileList(vector<Document> & documents, string fullPath, string baseDir, int dim, int languageCol, int fileNameCol, int featureNameCol, int featureValueCol) {
	ifstream inFile;
	inFile.open(fullPath.c_str());
	if (!inFile.is_open()) {
		cerr << "Unable to read file " << fullPath;
		exit(1);
	}
	string line;
	while (getline(inFile, line)) {
		vector<string> splitLine = splitString(line, ' ');
		int languageClass = getLanguageClass(splitLine[languageCol]);
		string filePath = baseDir+splitLine[fileNameCol];
		documents.push_back(readDocument(languageClass, filePath, dim, featureNameCol, featureValueCol));
	}
	inFile.close();
}
vector<Document> fetchDocumentsFromFileList(int speechSet, string fileListDir, string baseDir, int dim, bool limitFeature) {
	vector<Document> documents;
	int featureValueCol = 1;
	if (limitFeature) {
		featureValueCol = 2;
	}
	if (speechSet < 3) {//Normal CallFriend set
		fetchDocumentsFromFileList(documents, fileListDir+SET_INPUTFILE_NAMES[speechSet], baseDir, dim, 0, 1, 0, featureValueCol);
	}
	else if (speechSet == NISTSET) {//FileListDir is supposed to point directly to file
		fetchDocumentsFromFileList(documents, fileListDir, baseDir, dim, 4, 0, 0, featureValueCol);
	}
	else if (speechSet == TRAIN_AND_DEVSET) {
		fetchDocumentsFromFileList(documents, fileListDir+SET_INPUTFILE_NAMES[TRAINSET], baseDir, dim, 0, 1, 0, featureValueCol);
		fetchDocumentsFromFileList(documents, fileListDir+SET_INPUTFILE_NAMES[DEVSET], baseDir, dim, 0, 1, 0, featureValueCol);
	}		
	return documents;
}


void writeDocument(ofstream & outFile, Document & document, int dim) {
	outFile << document.languageClass;
	for (int i = 0; i < dim; i++) {
		outFile << " " << (i+1) << ":" << document.iVector[i];
	}
	outFile << "\n";
}

void writeDocuments(vector<Document> & documents, string fullPath, int dim) {
	ofstream outFile;
	outFile.open(fullPath.c_str());
	if (!outFile.is_open()) {
		cerr << "Unable to write to file " << fullPath;
		exit(1);
	}
	for (unsigned int i = 0; i < documents.size(); i++) {
		writeDocument(outFile, documents[i], dim);
	}
}


/*
//Deprecated
void readDocuments(int languageClass, string dirpath, vector<Document> & documents, int dim) {
	
	//OS specific (windows or unix-based assumed)
	#ifdef _WIN32
	WIN32_FIND_DATA FindData;
	HANDLE hFind;
	string searchString = dirpath+TRANS_SEARCH_PATTERN;
	hFind = FindFirstFile( searchString.c_str(), &FindData );

	if (hFind == INVALID_HANDLE_VALUE) {
		cerr << "Error finding files in directory " << dirpath;
		exit(1);
	}
	do {
		string filePath = dirpath+FindData.cFileName;
		documents.push_back(readDocument(languageClass, filePath, dim));
	} while (FindNextFile(hFind, &FindData) > 0);
	if (GetLastError() != ERROR_NO_MORE_FILES) {
		cerr << "Error while reading directory " << dirpath;
	}
	#else
	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(dirpath.c_str());
	//dirpath.c_str()????
	if (dpdf != NULL) {
		while (epdf = readdir(dpdf)) {
			string filepath = dirpath+string(epdf->d_name);
			if (filepath.find(TRANS_EXTENSION) > 0) {
				documents.push_back(readDocument(languageClass, filePath, dim));
			}
		}
		closedir (dpdf);
	} else {
		cerr << "Error while reading directory " << dirpath;
		exit(1);
	}
	#endif
}

//Deprecated
vector<Document> fetchDocuments(int speechSet, string speechPath, int dim) {
	//string speechPath = "c:\\test\\?S\\docnum30\\?L\\";//?S=one of speech sets, ?L=language
	vector<Document> documents;
	size_t temp;
	temp=speechPath.find("?S");
	if (temp != string::npos) {
		speechPath = speechPath.replace(temp, 2, SPEECH_SETS[speechSet]);
	}
	string dirPath = speechPath;
	temp = speechPath.find("?L");
	for (int i = 0; i < NUMBER_OF_LANGUAGES; i++) {
		if (temp != string::npos) {
			dirPath = speechPath.replace(temp, 2, LANGUAGES[i].first);
		}
		readDocuments(LANGUAGES[i].second, dirPath, documents, dim);
	}
	return documents;
}
*/




