#ifndef _HIERARCHICAL_SOFTMAX_H
#define _HIERARCHICAL_SOFTMAX_H

#include "data.h"
#include "sync_param.h"

#define PAIR_TYPE_NUM 4

typedef float Dtype;

typedef struct _NetParam {
  char *syn0_init_file, *syn1_init_file;
  int syn0_hid, syn1_hid;
  // Dtype *syn0, *syn1;
  Vocab *vocab;
  int layer1_size;
} NetParam;

typedef struct _LossArg {
  int pstart, pend;
  float loss[PAIR_TYPE_NUM];
  NetParam* param;
  DataPair* data;
} LossArg;

typedef struct _TrainArg {
  int pstart, pend;
  Dtype learning_rate;
  Dtype weight_decay;
  NetParam* param;
  DataPair* data;
} TrainArg;

void init_hs(NetParam *param);
void *compute_hs_loss(void *args);
void *train_hs(void *args);

#endif