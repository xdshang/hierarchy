/*
 * ValueDef.h
 *
 *  Created on: Oct 29, 2014
 *      Author: lms-gpu
 */

#ifndef VALUEDEF_H_
#define VALUEDEF_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <utility>
#include <string>
#include <iostream>
#include <algorithm>
#include <set>
#include <map>

#define MAX_STRING 100  //Define the longest word length.
#define EXP_TABLE_SIZE 1000  //Precompute the exp value and store in table.
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_PAIRS_NUM 15000000 // 500000//10000000
#define NUM_IMG  930324 // 456540// 930324//269648  //Num of Images.
#define NUM_NONZERO 741214820 //213660720 //318031140 //682022563//195650975  //num of Nonzeros
#define feat_size 4096// 468//500

struct vocab_word {
	long long cn;  // the count of the vocabulary word.
	int *point;
	char *word, *code, codelen;
};
struct thread_par{
	int thread_id;
	int mini_index;
	int mini_pairsno;
	float *neu1e;
	float *syn0_temp;
};
struct thread_loss{
	float *loss_thread;
	float *aa_loss_thread;
	float *ab_loss_thread;
	float *ba_loss_thread;
	float *bb_loss_thread;
	int thread_id;
};

extern const int vocab_hash_size;  // Maximum 30 * 0.7 = 21M words in the vocabulary
extern const int table_size;
extern char train_file[MAX_STRING], syn0_best_file[MAX_STRING], syn1_best_file[MAX_STRING], feat_file[MAX_STRING], feat_folder[MAX_STRING], syn1_delta_file[MAX_STRING], syn0_delta_file[MAX_STRING],
image_index_file[MAX_STRING], save_vocab_file[MAX_STRING], w_best_file[MAX_STRING], read_vocab_file[MAX_STRING], iter_w_best_file[MAX_STRING], iter_w_delta_best_file[MAX_STRING],
iter_syn0_best_file[MAX_STRING], iter_syn0_delta_best_file[MAX_STRING], iter_syn1_best_file[MAX_STRING], iter_syn1_delta_best_file[MAX_STRING], syn0_temp_file[MAX_STRING], tree_file[MAX_STRING];
extern long long vocab_max_size, vocab_size, pair_size;
extern long long train_words, pair_count_actual, file_size, total_pairs;
extern float *syn0, *syn1, *expTable, *syn0_delta, *syn1_delta, *syn1neg, *tranMatrix_delta, *syn1_update, *syn0_update, *tranMatrix_update, *tranMatrix_best, *tranMatrix_delta_best, *syn0_best, *syn1_best, *syn1_delta_best, *syn0_delta_best;
extern float tranMatrix[1000 * feat_size];
extern float sample, max_norm, iter_loss, fixedw_alpha_best, fixsy_alpha_best, iter_loss_best;
extern int binary, debug_mode, min_count, num_threads, min_reduce, num_epoches, batch_size, scale, layer1_size, offset_sy, offset_w, iteration, MaxIteration;
extern int *vocab_hash, *table;

extern struct vocab_word *vocab;
extern std::pair<char[MAX_STRING], char[MAX_STRING]> pairs[MAX_PAIRS_NUM];
extern std::map<std::string, int> image_index;
extern std::vector <int> randomVector;
extern int* nonzeroIdx; //nonzeroIdx[i] is the starting idx of array Idx
extern int* nonzeroLen; //nonzeroLen[i] is the #non zero entries of image i.
extern float* Val;
extern int* Idx;

#endif /* VALUEDEF_H_ */

