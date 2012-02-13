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

void trainiVectors(string inFileList, string baseDir, string outLoc, int height, int width, unsigned int seed, bool limitFeature, int threads);

#endif