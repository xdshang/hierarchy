/*
 * Common.h
 *
 *  Created on: Oct 29, 2014
 *      Author: lms-gpu
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <iomanip>
#include <cmath>
#include <map>
#include <time.h>
#include <vector>
#include <algorithm>
using namespace std;

#include "ValueDef.h"

//Self Writtern and Common.
void countNonzero (char featFilePath[MAX_STRING]);
void countNonzero_folder(char * featfolder);
void readFeat(char featFilePath[MAX_STRING]);
void readFeat_folder(char* featFolder);
void readImageIndex (char imageIndexPath[MAX_STRING]);
void LearnPairsFromFile ();
void ReadSy_Delta();
void ResetPara_sy();
void ResetPara_w();
void InitSyn_best();
void ResetDelta();
bool endsWith (char* base, const char* str);

#endif /* COMMON_H_ */
