#ifndef IVECTIO_H
#define IVECTIO_H
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "document.h"
#include "FeatureSpace.h"
#include "Configuration.h"
#include <vector>

const static int TRAINSET = 0;
const static int DEVSET = 1;
const static int EVLSET = 2;
const static int STRAINSET = 3;

std::vector<Document> fetchDocumentsFromFileList(int speechSet, Configuration &config);
void fetchDocumentsFromFileList(std::vector<Document> & documents, std::string fullPath, std::string baseDir, int dim, int languageCol, int fileNameCol, int featureNameCol, int featureValueCol);
//vector<Document> fetchDocuments(int speechSet, string speechPath, int dim);
void writeDocuments(std::vector<Document> & documents, std::string fullPath);
FeatureSpace readSpace(Configuration & config);
void writeSpace(FeatureSpace & space, std::string fullPath);

#endif
