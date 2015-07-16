#ifndef _HIERARCHICAL_SOFTMAX_H
#define _HIERARCHICAL_SOFTMAX_H

#include "data.h"

#define PAIR_TYPE_NUM 4

typedef float Dtype;

typedef struct _NetParam {
  char *syn0_init_file, *syn1_init_file;
  Dtype *syn0, *syn1;
  Dtype *syn0_delta, *syn1_delta;
  Dtype *syn0_best, *syn1_best;
  Vocab *vocab;
  size_t layer1_size;
} NetParam;

typedef struct _LossArg {
  int thread_num, thread_id;
  float loss[PAIR_TYPE_NUM];
  NetParam* param;
  DataPair* data;
} LossArg;

typedef struct _TrainArg {
  int thread_num, thread_id;
  Dtype learning_rate;
  Dtype weight_decay;
  Dtype momentum;
  NetParam* param;
  DataPair* data;
} TrainArg;

void init_hs(NetParam *param);
void *compute_hs_loss(void *args);
void *train_hs(void *args);

#endif