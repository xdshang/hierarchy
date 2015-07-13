/*
 * ValueDef1.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: lms-gpu
 */
#include "ValueDef.h"

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

using namespace std;

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int table_size = 1e8;
char train_file[MAX_STRING], syn0_best_file[MAX_STRING], syn1_best_file[MAX_STRING], feat_file[MAX_STRING],feat_folder[MAX_STRING], syn1_delta_file[MAX_STRING], syn0_delta_file[MAX_STRING],
image_index_file[MAX_STRING], save_vocab_file[MAX_STRING], w_best_file[MAX_STRING], read_vocab_file[MAX_STRING],iter_w_best_file[MAX_STRING], iter_w_delta_best_file[MAX_STRING],
iter_syn0_best_file[MAX_STRING], iter_syn0_delta_best_file[MAX_STRING], iter_syn1_best_file[MAX_STRING], iter_syn1_delta_best_file[MAX_STRING],syn0_temp_file[MAX_STRING], tree_file[MAX_STRING];
long long vocab_max_size = 1000, vocab_size = 0, pair_size;
long long train_words = 0, pair_count_actual = 0, file_size = 0, total_pairs = 0;
float *syn0, *syn1, *expTable, *syn0_delta, *syn1_delta, *syn1neg,
*tranMatrix_delta, *syn1_update, *syn0_update, *tranMatrix_update,*tranMatrix_best, *tranMatrix_delta_best, *syn0_best, *syn1_best, *syn1_delta_best, *syn0_delta_best;
float tranMatrix[1000 * feat_size];
float sample = 0, max_norm = 1.0, iter_loss = 0, fixedw_alpha_best, fixsy_alpha_best, iter_loss_best;
int binary = 0, debug_mode = 2, min_count = 5, num_threads = 1, min_reduce = 1, num_epoches = 1,
		batch_size = 512, scale = 1, layer1_size = 1000, offset_sy, offset_w, iteration = 0,MaxIteration = 100;
int *vocab_hash, *table;
int offset = 0;

struct vocab_word *vocab;
pair<char[MAX_STRING], char[MAX_STRING]> pairs[MAX_PAIRS_NUM];
map<string, int> image_index;
std::vector <int> randomVector;
