/*
 * Word2Vec.h
 *
 *  Created on: Oct 29, 2014
 *      Author: lms-gpu
 */

/*
 * Define Functions from Word2Vec
 */

#ifndef WORD2VEC_H_
#define WORD2VEC_H_
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

//Word2Vec
int ArgPos(char *str, int argc, char **argv);
int VocabCompare(const void *a, const void *b);
int GetWordHash(const char *word) ;
int SearchVocab(const char *word);
void ReadWord(char *word, FILE *fin);
int ReadWordIndex(FILE *fin);
void SortVocab();
int AddWordToVocab(char *word);
void SaveVocab();
void ReadVocab();
void CreateBinaryTree();
void ReduceVocab();
void LearnVocabFromTrainFile();
void BuildPara();

#endif /* WORD2VEC_H_ */
