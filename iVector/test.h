#ifndef TEST_H
#define TEST_H
#include "iVectMath.h"
#include "log.h"
#include "Document.h"
#include "FeatureSpace.h"
#include <iostream>
#include "iVectIO.h"

//Only for unix based
#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace std;

void testAll(int width);

#endif