#ifndef _DATA_H
#define _DATA_H

#include <stddef.h>

typedef struct _Pair {
  char *first, *second;
} Pair;

typedef struct _DataPair {
  Pair* data;
  size_t size;
  size_t capacity;
} DataPair;

typedef struct _VocabWord {
  int cn;  // the count of the vocabulary word.
  int *point;
  char *word, *code, codelen;
} VocabWord;

typedef struct _Vocab {
  VocabWord* data;
  size_t size;
  size_t capacity;
} Vocab;

typedef struct _DataParam {
  char* train_file;
  char* read_vocab_file;
  char* save_vocab_file;
  int min_count;
} DataParam;

void init_data(const DataParam* param, Vocab* vocab, DataPair* pairs);
int search_vocab(const Vocab* vocab, const char *word);

#endif