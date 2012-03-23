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
#include "configuration.h"
#include <vector>

void shortTrainiVectors(Configuration config);
void trainiVectors(Configuration config);

#endif