#ifndef TEST_H
#define TEST_H
#include "iVectMath.h"
#include "log.h"
#include "Document.h"
#include "FeatureSpace.h"
#include <iostream>
#include "iVectIO.h"
#include "iVectThread.h"
#include <vector>
#include "Configuration.h"

//Only for unix based
#ifndef _WIN32
#include <sys/time.h>
#endif

void testAll(Configuration config);

#endif