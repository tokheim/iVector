#ifndef IVECTIO_H
#define IVECTIO_H
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "document.h"


/*
//DEPRECATED
//OS specific (windows or unix-based assumed)
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif
*/

using namespace std;

const static int TRAINSET = 0;
const static int DEVSET = 1;
const static int EVLSET = 2;
const static int NISTSET = 3;
const static int TRAIN_AND_DEVSET = 4;

vector<Document> fetchDocumentsFromFileList(int speechSet, string fileListDir, string baseDir, int dim, bool limitFeature);
void fetchDocumentsFromFileList(vector<Document> & documents, string fullPath, string baseDir, int dim, int languageCol, int fileNameCol, int featureNameCol, int featureValueCol);
//vector<Document> fetchDocuments(int speechSet, string speechPath, int dim);
void writeDocuments(vector<Document> & documents, string fullPath);

#endif
