#include "hierarchical_softmax.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#define MAX_CODE_LENGTH 40
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

static Dtype *expTable = NULL;

void create_binary_tree(Vocab *vocab);

void init_hs(NetParam *param) {
  param->syn0_hid = create_sync_param(param->syn0_init_file, 
      param->vocab->size, param->layer1_size);
  assert(param->syn0_hid >= 0);

  param->syn1_hid = create_sync_param(param->syn1_init_file, 
      param->vocab->size, param->layer1_size);
  assert(param->syn1_hid >= 0);

  create_binary_tree(param->vocab);

  if (!expTable) {
    int i;
    expTable = (Dtype*)malloc((EXP_TABLE_SIZE + 1) * sizeof(Dtype));
    for (i = 0; i < EXP_TABLE_SIZE + 1; ++i) {
      // Precompute the exp() table
      expTable[i] = exp((i / (Dtype)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
      // Precompute f(x) = x / (x + 1)
      expTable[i] = expTable[i] / (expTable[i] + 1);
    }
  }
}

void *compute_hs_loss(void *args) {
  int m, c, d, pair_stamp, word_A[PAIR_TYPE_NUM], word_B[PAIR_TYPE_NUM];
  LossArg* thread_arg = (LossArg*)args;
  NetParam* param = thread_arg->param;
  DataPair* pairs = thread_arg->data;
  Dtype f;
  const Dtype *syn0 = NULL, *syn1 = NULL;

  for (m = 0; m < PAIR_TYPE_NUM; ++m) {
    thread_arg->loss[m] = 0;
  }

  for (pair_stamp = thread_arg->pstart; pair_stamp < thread_arg->pend; ++pair_stamp) {
    word_A[0] = search_vocab(param->vocab, pairs->data[pair_stamp].first);
    word_B[0] = search_vocab(param->vocab, pairs->data[pair_stamp].second);
    assert(word_A[0] >= 0);
    assert(word_B[0] >= 0);
    word_A[1] = word_B[0];
    word_B[1] = word_A[0];
    word_A[2] = word_A[0];
    word_B[2] = word_A[0];
    word_A[3] = word_B[0];
    word_B[3] = word_B[0];

    for (m = 0; m < PAIR_TYPE_NUM; ++m) {
      syn0 = sync_param(param->syn0_hid, word_A[m]);
      for (d = 0; d < param->vocab->data[word_B[m]].codelen; ++d) {
        f = 0;
        syn1 = sync_param(param->syn1_hid, param->vocab->data[word_B[m]].point[d]);
        // Propagate hidden -> output
        for (c = 0; c < param->layer1_size; ++c) {
          f += syn0[c] * syn1[c];  
        }
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else {
          f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          if (param->vocab->data[word_B[m]].code[d] == 0) {
            thread_arg->loss[m] -= log(f);
          }
          else {
            thread_arg->loss[m] -= log(1 - f);
          }
        }
      }
    }
  }

  pthread_exit(NULL);
}

void* train_hs(void *args) {
  int m, c, d, pair_stamp, word_A[PAIR_TYPE_NUM], word_B[PAIR_TYPE_NUM];
  TrainArg* thread_arg = (TrainArg*)args;
  NetParam* param = thread_arg->param;
  DataPair* pairs = thread_arg->data;
  Dtype f, g, *syn1;
  const Dtype *syn0;
  // Dtype *neu1e = (Dtype*)malloc(param->layer1_size * sizeof(Dtype));

  for (pair_stamp = thread_arg->pstart; pair_stamp < thread_arg->pend; ++pair_stamp) {
    word_A[0] = search_vocab(param->vocab, pairs->data[pair_stamp].first);
    word_B[0] = search_vocab(param->vocab, pairs->data[pair_stamp].second);
    assert(word_A[0] >= 0);
    assert(word_B[0] >= 0);
    word_A[1] = word_B[0];
    word_B[1] = word_A[0];
    word_A[2] = word_A[0];
    word_B[2] = word_A[0];
    word_A[3] = word_B[0];
    word_B[3] = word_B[0];

    for (m = 0; m < PAIR_TYPE_NUM; ++m) {
      // memset(neu1e, 0, param->layer1_size * sizeof(Dtype));
      syn0 = sync_param(param->syn0_hid, word_A[m]);
      // HIERARCHICAL SOFTMAX
      for (d = 0; d < param->vocab->data[word_B[m]].codelen; ++d) {
        f = 0;
        syn1 = mutable_sync_param(param->syn1_hid, param->vocab->data[word_B[m]].point[d]);
        // Propagate hidden -> output
        for (c = 0; c < param->layer1_size; ++c) {
          f += syn0[c] * syn1[c];  
        }
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else {
          f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        }
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - param->vocab->data[word_B[m]].code[d] - f) * thread_arg->learning_rate;
        // Propagate errors output -> hidden
        // for (c = 0; c < param->layer1_size; ++c) {
        //   neu1e[c] += g * param->syn1[c + node_idx];
        // }
        // Learn weights hidden -> output
        for (c = 0; c < param->layer1_size; ++c) {
          syn1[c] += g * syn0[c] - thread_arg->weight_decay * thread_arg->learning_rate * syn1[c];
        }
      }
    }
    //Compute mini batch loss and add to the neu1e.  // for (c = 0; c < layer1_size; c++){
    //  syn0_delta[c + index_A] = 0.9 * syn0_delta[c + index_A] - 0.0001 * alpha * syn0[c + index_A] + neu1e[c];
    //  syn0[c + index_A] += syn0_delta[c + index_A];
    // }
  }

  // free(neu1e);
  pthread_exit(NULL);
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void create_binary_tree(Vocab *vocab) {
  int a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  char *binary = (char*)calloc(vocab->size * 2 + 1, sizeof(char));
  int *count = (int*)malloc((vocab->size * 2 + 1) * sizeof(int));
  int *parent_node = (int*)malloc((vocab->size * 2 + 1) * sizeof(int));

  for (a = 0; a < vocab->size; a++) {
    count[a] = vocab->data[a].cn;
  }
  for (a = vocab->size; a < vocab->size * 2; a++) {
    count[a] = INT_MAX;
  }
  pos1 = vocab->size - 1;
  pos2 = vocab->size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab->size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab->size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab->size + a;
    parent_node[min2i] = vocab->size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab->size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab->size * 2 - 2) break;
    }
    assert(i < MAX_CODE_LENGTH);
    vocab->data[a].codelen = i;
    vocab->data[a].point = (int*)malloc(i * sizeof(int));
    vocab->data[a].code = (char*)malloc(i * sizeof(char));
    vocab->data[a].point[0] = vocab->size - 2;
    for (b = 0; b < i; b++) {
      vocab->data[a].code[i - b - 1] = code[b];
      vocab->data[a].point[i - b] = point[b] - vocab->size;
    }
  }

  free(count);
  free(binary);
  free(parent_node);
}