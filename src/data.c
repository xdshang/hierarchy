#include "data.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_STRING 100 
#define MAX_PAIRS_NUM 15000000

const int vocab_hash_size = 20000000;
int *vocab_hash;

void learn_vocab_from_train_file(const DataParam* param, Vocab* vocab);
void learn_pairs_from_file(const DataParam* param, DataPair* pairs);
void read_vocab(const DataParam* param, Vocab* vocab);
void save_vocab(const DataParam* param, const Vocab* vocab);

void init_data(const DataParam* param, Vocab* vocab, DataPair* pairs) {
  vocab_hash = (int*)malloc(vocab_hash_size * sizeof(int));
  memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
  vocab->size = 0;
  vocab->capacity = 1000;
  vocab->data = (VocabWord*)malloc(vocab->capacity * sizeof(VocabWord));
  pairs->size = 0;
  pairs->capacity = 1000;
  pairs->data = (Pair*)malloc(pairs->capacity * sizeof(Pair));
  if (param->read_vocab_file) {
    read_vocab(param, vocab); 
  }
  else {
    learn_vocab_from_train_file(param, vocab);
  }
  if (param->save_vocab_file) {
    save_vocab(param, vocab);
  }
  learn_pairs_from_file(param, pairs);
}

// Returns hash value of a word   open addressing linear probing
unsigned int get_word_hash(const char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a]; // based on the word composition to generate the hash value. some words may get the same hash value and use the linear probing to solve the problem.
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int search_vocab(const Vocab* vocab, const char *word) {
  unsigned int hash = get_word_hash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab->data[vocab_hash[hash]].word)) return vocab_hash[hash];  //return the hash index of the word in the vocabulary
    hash = (hash + 1) % vocab_hash_size;  // if word hash conflicts occurs, the hash will be added 1 and search again.
  }
  return -1;
}

// Adds a word to the vocabulary and revise the vocab_hash table.
int add_word_to_vocab(Vocab* vocab, const char *word) {
  unsigned int hash;
  int length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab->data[vocab->size].word = (char*)malloc(length * sizeof(char));
  strncpy(vocab->data[vocab->size].word, word, length);
  vocab->data[vocab->size].cn = 0;
  vocab->size++;
  // Reallocate memory if needed
  if (vocab->size + 2 >= vocab->capacity) {
    vocab->capacity += 1000;
    vocab->data = (VocabWord*)realloc(vocab->data, vocab->capacity * sizeof(VocabWord));
  }
  hash = get_word_hash(word);
  while (vocab_hash[hash] != -1) {
    hash = (hash + 1) % vocab_hash_size;
  }
  vocab_hash[hash] = vocab->size - 1;
  return vocab->size - 1;
}

// Reduces the vocabulary by removing infrequent tokens
void reduce_vocab(Vocab* vocab) {
  static int min_reduce = 1;
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab->size; a++) {
    if (vocab->data[a].cn > min_reduce) {
      vocab->data[b].cn = vocab->data[a].cn;
      vocab->data[b].word = vocab->data[a].word;
      b++;
    } 
    else {
      free(vocab->data[a].word);
    }
  }
  vocab->size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab->size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = get_word_hash(vocab->data[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  min_reduce++;
}

// Sorts the vocabulary by frequency using word counts
int vocab_compare(const void *a, const void *b) {
  return ((VocabWord*)b)->cn - ((VocabWord*)a)->cn;
}
void sort_vocab(Vocab* vocab, const int min_count) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(vocab->data, vocab->size, sizeof(VocabWord), vocab_compare);
  memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
  size = vocab->size;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab->data[a].cn < min_count) {
      vocab->size--;
      free(vocab->data[vocab->size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = get_word_hash(vocab->data[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
    }
  }
  vocab->data = (VocabWord*)realloc(vocab->data, (vocab->size + 1) * sizeof(VocabWord));
  vocab->capacity = vocab->size + 1;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void read_word(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin); // read the character that the internal file pointer, after read it, the internal file pointer will be moved to next automaitically.
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

//Learn Vocabulary from Train File.
void learn_vocab_from_train_file(const DataParam* param, Vocab* vocab) {
  char word[MAX_STRING];
  FILE *fin;
  int a, i;
  long long train_words = 0;
  fin = fopen(param->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab->size = 0;
  //  AddWordToVocab((char *)"</s>");
  while (1) {
    read_word(word, fin);
    if (feof(fin)) break;  // file end, break out.
    if (strcmp(word, (const char*)"</s>") == 0)
      continue;
    train_words++;
    if (train_words % 100000 == 0) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = search_vocab(vocab, word);
    if (i == -1) {
      a = add_word_to_vocab(vocab, word);
      vocab->data[a].cn = 1;
    } 
    else {
      vocab->data[i].cn++;
    }
    if (vocab->size > vocab_hash_size * 0.7) {
      printf("WARNING: vocab size is too big for hash, removing infrequent words.\n");
      reduce_vocab(vocab);
    }
  }
  sort_vocab(vocab, param->min_count); //It has min count settings.

  printf("Vocab size: %d\n", vocab->size);
  printf("Words in train file: %lld\n", train_words);

  fclose(fin);
}

void read_vocab(const DataParam* param, Vocab* vocab) {
  int a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(param->read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
  vocab->size = 0;
  while (1) {
    read_word(word, fin);
    if (feof(fin)) break;
    a = add_word_to_vocab(vocab, word);
    fscanf(fin, "%d%c", &(vocab->data[a].cn), &c);
    i++;
  }
  sort_vocab(vocab, param->min_count);
  printf("Vocab size: %d\n", vocab->size);
  fin = fopen(param->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  fclose(fin);
}

void save_vocab(const DataParam* param, const Vocab* vocab) {
  int i;
  FILE *fo = fopen(param->save_vocab_file, "wb");
  for (i = 0; i < vocab->size; i++) {
    fprintf(fo, "%s %d\n", vocab->data[i].word, vocab->data[i].cn);
  }
  fclose(fo);
}

void learn_pairs_from_file(const DataParam* param, DataPair* pairs) {
  FILE *fin;
  char word[MAX_STRING];
  fin = fopen(param->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  pairs->size = 0;
  int wordstamp = 0;
  while (1) {
    read_word(word, fin);
    if (feof(fin)) break;  // file end, break out.
    if(strcmp(word, (char *)"</s>") == 0)
      continue;
    if (pairs->size + 2 >= pairs->capacity) {
      pairs->capacity += 1000;
      pairs->data = (Pair*)realloc(pairs->data, pairs->capacity * sizeof(Pair));
    }
    if (wordstamp % 2 == 0) {
      pairs->data[pairs->size].first = (char*)malloc((strlen(word) + 1) * sizeof(char));
      strcpy(pairs->data[pairs->size].first, word);
    }
    else {
      pairs->data[pairs->size].second = (char*)malloc((strlen(word) + 1) * sizeof(char));
      strcpy(pairs->data[pairs->size].second, word);
      pairs->size++;
    }
    wordstamp++;
  }
  pairs->data = (Pair*)realloc(pairs->data, (pairs->size + 1) * sizeof(Pair));
  pairs->capacity = pairs->size + 1;
  fclose(fin);
}