#ifndef IVECTTRAIN_H
#define IVECTTRAIN_H
#include <string>
#include <iostream>
#include <math.h>
#include "FeatureSpace.h"
#include "Document.h"
#include "iVectMath.h"
#include "iVectIO.h"
#include "iVectThread.h"
#include "log.h"
#include <vector>

void trainiVectors(std::string inFileList, std::string baseDir, std::string outLoc, int height, int width, unsigned int seed, bool limitFeature, int threads);

#endif