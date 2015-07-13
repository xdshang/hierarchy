/*
 * PureVector_indep_loss.cpp
 *
 *  Created on: Oct 30, 2014
 *      Author: lms-gpu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <cassert>

#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;
/*
 * Currently,
 * 1. We change the input of the program: From sentences to the <image, label> pairs
 * 2. We also make a trick that, for softmax, we need the intermedia thing to connect two things, so we add <A, A> <B, B> pairs.
 * 3. We try to add the transformation matrix, if new image comes, we can transform the features into the same space with the label, or even users.
 * 4. We try to incoporate the program into Caffe to make it more fast.
 * 5. We delete the Multi-threads, in order to speed up, we use GPU.
 */

#define MAX_STRING 100  //Define the longest word length.
#define EXP_TABLE_SIZE 1000  //Precompute the exp value and store in table.
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_PAIRS_NUM 15000000

#define MOMENTUM 0.9
#define DECAY 0.1

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers
struct vocab_word {
	int cn;  // the count of the vocabulary word.
	int *point;
	char *word, *code, codelen;
};
struct image_word{
	int cn;
	char *word;
};
struct tag_word{
	int cn;
	char *word;
};

struct thread_par{
	int* thread_id;
	real alpha;
};

struct negative_pairs{
	std::pair <char[MAX_STRING], char[MAX_STRING]> pair;
	long long max_degree_multi;
};

struct thread_loss{
	float *loss_thread;
	float *aa_loss_thread;
	float *ab_loss_thread;
	float *ba_loss_thread;
	float *bb_loss_thread;
	int thread_id;
};

char train_file[MAX_STRING], syn0_best_file[MAX_STRING], syn1_best_file[MAX_STRING], syn1_curr_file[MAX_STRING], 
syn0_delta_file[MAX_STRING],  syn1_delta_file[MAX_STRING],
save_vocab_file[MAX_STRING], syn0_init_file[MAX_STRING], read_vocab_file[MAX_STRING], tree_file[MAX_STRING];

struct vocab_word *vocab;
struct image_word *image_words;
struct tag_word *tag_words;
struct negative_pairs *neg_pairs;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, num_epoches = 1;
int *vocab_hash;
float negative_ratio = 0.1;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100, total_pairs = 0;
long long train_words = 0, pair_count_actual = 0, file_size = 0, classes = 0;

real alpha = 0.001, starting_alpha, sample = 0;
real *syn0, *syn1, *expTable, *syn0_delta, *syn1_delta, *syn0_best, *syn1neg;
clock_t start;
pair<char[MAX_STRING], char[MAX_STRING]> *pairs;
const int table_size = 1e8;
int *table;
long long pair_size; //number of pairs;
long long negative_pair_size; //record which has no such pairs but max degree. // real data is <image, user> pairs.
double epoch_loss = 0.0 ;

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Returns hash value of a word   open addressing linear probing
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a]; // based on the word composition to generate the hash value. some words may get the same hash value and use the linear probing to solve the problem.
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];  //return the hash index of the word in the vocabulary
		hash = (hash + 1) % vocab_hash_size;  // if word hash conflicts occurs, the hash will be added 1 and search again.
	}
	return -1;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
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
// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;  // if end of the file.
	return SearchVocab(word);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[0], vocab_size, sizeof(struct vocab_word), VocabCompare);  // sort the words by count from large to small?
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			vocab_size--;
			free(vocab[vocab_size].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash=GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}
/** detecting whether base is ends with str
 */
bool endsWith (char* base, const char* str) {
	int blen = strlen(base);
	int slen = strlen(str);
	return (blen >= slen) && (0 == strcmp(base + blen - slen, str));
}

/*
 * find the max multiplication of two nodes degree.
 */
void FindMaxDegreeMultiply(){
	long long a,b,c;
	b=0; c=0;
	image_words = (struct image_word *)realloc(image_words, (vocab_size + 1) * sizeof(struct image_word));
	tag_words = (struct tag_word *)realloc(tag_words, (vocab_size + 1) * sizeof(struct tag_word));

	for(a = 0; a < vocab_size; a ++){
		if(endsWith(vocab[a].word, ".jpg")){
			image_words[b].cn = vocab[a].cn;
			image_words[b].word = vocab[a].word;
			b++;
		}
		else{
			tag_words[c].cn = vocab[a].cn;
			tag_words[c].word = vocab[a].word;
			c++;
		}
	}

	std::set<string> pair_set;
	string s1, s2, s1_2;

	for (a = 0; a < pair_size; a ++){
		s1 = string(pairs[a].first);
		s2 = string(pairs[a].second);

		s1_2 ="\t";
		pair_set.insert(s1 + s1_2 + s2);
	}

	//possible negative pairs
	long long temp_negative_pair_size = 0;
	struct negative_pairs *temp_neg_pairs;
	temp_negative_pair_size = pair_size * (negative_ratio + 0.1);
	temp_neg_pairs = (struct negative_pairs *) calloc(temp_negative_pair_size + 1, sizeof(struct negative_pairs)); //allocate spaces for negative pairs.
	cout << "temp negative_pair_size \t" <<temp_negative_pair_size<<endl;

	long long p = 0, q = 0;
	bool tag = false;
	pair<string[MAX_STRING], string[MAX_STRING]> possible_neg_pair;
	a = 0;
	while (a < temp_negative_pair_size){
		for (p = 0; p < b; p++){  //image_index
			for (q = 0; q < c; q++){  //user_index
				if (tag_words[q].cn < image_words[p].cn){  //check user degree > image degree.
					break;
				}
				else{
					if (!tag){
						//add negative pairs into the negative pairs
						strncpy(temp_neg_pairs[a].pair.first, image_words[p].word, MAX_STRING);
						strncpy(temp_neg_pairs[a].pair.second, tag_words[q].word, MAX_STRING);
						//						cout << neg_pairs[a].pair.first  << "\t" << neg_pairs[a].pair.second << endl;
						temp_neg_pairs[a].max_degree_multi = image_words[p].cn * tag_words[q].cn;
						a++;
					}
				}
				if(a >= temp_negative_pair_size) break;
			}
			if(a >= temp_negative_pair_size) break;
		}
	}

	negative_pair_size = pair_size * negative_ratio;
	neg_pairs = (struct negative_pairs *) realloc(neg_pairs, (negative_pair_size + 1) * sizeof(struct negative_pairs)); //allocate spaces for negative pairs.
	cout << "negative_pair_size \t" << negative_pair_size << endl;

	//delete occured.
	string temp;
	bool is_in = true;
	b = 0;
	a = 0;
	while (b < negative_pair_size){
		temp = string(temp_neg_pairs[a].pair.first) + string("\t") + string(temp_neg_pairs[a].pair.second);
//		cout << "temp string: "<< temp <<endl;
		is_in = pair_set.find(temp) != pair_set.end();

		if(!is_in){
			strncpy(neg_pairs[b].pair.first, temp_neg_pairs[a].pair.first, MAX_STRING);
			strncpy(neg_pairs[b].pair.second, temp_neg_pairs[a].pair.second, MAX_STRING);
			neg_pairs[b].max_degree_multi = temp_neg_pairs[a].max_degree_multi;
			b++;
		}else{
//			cout << "occur \t";
//			cout << temp_neg_pairs[a].pair.first  << "\t" << temp_neg_pairs[a].pair.second << endl;
		}
		a++;
	}

	free (temp_neg_pairs);
	cout << neg_pairs[negative_pair_size - 1 ].pair.first  << "\t" << neg_pairs[negative_pair_size - 1].pair.second <<  endl;
	cout << neg_pairs[negative_pair_size - 1].max_degree_multi <<endl;
}

void LearnPairsFromFile ()
{
	FILE *fin;
	char word[MAX_STRING];
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	long long wordstamp = 0;
	string ends = "jpg";

	try{
		while (1) {
			ReadWord(word, fin);
			if (feof(fin)) break;  // file end, break out.
			if(strcmp(word, (char *)"</s>") == 0)
				continue;
			wordstamp = wordstamp + 1;
			if (wordstamp % 2 == 1){
				strcpy(pairs[wordstamp/2].first, word);
			}
			else{
				strcpy(pairs[wordstamp/2 - 1].second, word); //tags
				pair_size = pair_size + 1;
				if (endsWith (pairs[wordstamp/2 - 1].second, ends.c_str())){
					total_pairs = total_pairs + 1;
				}else{
					total_pairs = total_pairs + 4;
				}
			}
		}
	}
	catch(...){
		cout << wordstamp <<endl;
	}
	fclose(fin);
}
// Adds a word to the vocabulary and revise the vocab_hash table.
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}
void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;

		a = AddWordToVocab(word);
		fscanf(fin, "%d%c", &vocab[a].cn, &c);
		i++;
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	//	for (a = 0; a < vocab_size; a++) count[a] = 1;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
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
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
  // save tree to file
  ofstream fout(tree_file);
  for(a = 0; a < vocab_size; ++a){
    fout << vocab[a].word << ' ' << (int)vocab[a].codelen;
    for(b = 0; b < vocab[a].codelen; ++b)
      fout << ' ' << (int)vocab[a].code[b];
    for(b = 0; b < vocab[a].codelen; ++b)
      fout << ' ' << vocab[a].point[b];
    fout << endl;
  }
  fout.close();
  cout << "tree saved to file." << endl;
	free(count);
	free(binary);
	free(parent_node);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	} else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

//Learn Vocabulary from Train File.
void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	//	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;  // file end, break out.
		if(strcmp(word, (char *)"</s>") == 0)
			continue;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab(); //It has min count settings.

	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void *TrainModelThread(void *id) {
	long long c, d, word_A, word_B, index_A, index_B, node_idx;
	long long pair_count = 0;
	real f, g;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));

	pair <char[MAX_STRING] , char[MAX_STRING]> image_user_pair; //Define a image-user pair.
	long long pair_stamp = 0;
	pair_stamp = pair_size / (long long)num_threads  * (long long)id;
	string ends = "jpg";
	while(1){
		strncpy(image_user_pair.first, pairs[pair_stamp].first, MAX_STRING);
		strncpy(image_user_pair.second, pairs[pair_stamp].second, MAX_STRING);
		if (pair_count >  pair_size/ num_threads) break;
		pair_count ++;

		//index of A's in vocabulary
		word_A = SearchVocab(image_user_pair.first);
		if(word_A == -1) {
			break;
		}
		//index of B's in vocabulary
		word_B = SearchVocab(image_user_pair.second);
		if(word_B == -1) {
			break;
		}

		//index of A in the syn0
		index_A = word_A * layer1_size;

		//index of B in the syn0
		index_B = word_B * layer1_size;

		//<A, B>
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word_B].codelen; d++) {
			f = 0;
			node_idx = vocab[word_B].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + index_A] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			}

			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word_B].code[d] - f) * alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++)
				neu1e[c] += g * syn1[c + node_idx];

			// Learn weights hidden -> output
			for(c=0; c<layer1_size; c++){
				syn1_delta[c + node_idx] =  MOMENTUM * syn1_delta[c + node_idx] - DECAY * alpha * syn1[c + node_idx] + g * syn0[c + index_A];
				syn1[c + node_idx] += syn1_delta[c + node_idx];
			}
		}

		//Compute mini batch loss and add to the neu1e.  // for (c = 0; c < layer1_size; c++){
		// 	syn0_delta[c + index_A] = 0.9 * syn0_delta[c + index_A] - 0.0001 * alpha * syn0[c + index_A] + neu1e[c];
		// 	syn0[c + index_A] += syn0_delta[c + index_A];
		// }

		// if (!endsWith (image_user_pair.second, ends.c_str())){
			//<B, A>
			for (c = 0; c < layer1_size; c++)
				neu1e[c] = 0;
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_A].codelen; d++) {
				f = 0;
				node_idx = vocab[word_A].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_B] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierachical node and changed. Here l2 is for the specific word.
				//			cout << "pair probability:  "<< pair_probablity <<endl;

				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				}
				// 'g' is the gradient multiplied by the learning rate
				g = (1 - vocab[word_A].code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < layer1_size; c++)
					neu1e[c] += g * syn1[c + node_idx];

				// Learn weights hidden -> output
				for(c=0; c<layer1_size; c++){
					syn1_delta[c + node_idx] = MOMENTUM * syn1_delta[c + node_idx] - DECAY * alpha * syn1[c + node_idx] + g * syn0[c + index_B];
					syn1[c + node_idx] += syn1_delta[c + node_idx];
				}
			}

			// for (c = 0; c < layer1_size; c++){
			// 	syn0_delta[c + index_B] = 0.9 * syn0_delta[c + index_B] - 0.0001 * alpha * syn0[c + index_B] + neu1e[c];
			// 	syn0[c + index_B] += syn0_delta[c + index_B];
			// }

			//<A, A>
			for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_A].codelen; d++) {
				f = 0;
				node_idx = vocab[word_A].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_A] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				}

				// 'g' is the gradient multiplied by the learning rate
				g = (1 - vocab[word_A].code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < layer1_size; c++)
					neu1e[c] += g * syn1[c + node_idx];

				// Learn weights hidden -> output
				for(c=0; c<layer1_size; c++){
					syn1_delta[c + node_idx] = MOMENTUM * syn1_delta[c + node_idx]- DECAY * alpha * syn1[c + node_idx] + g * syn0[c + index_A];
					syn1[c + node_idx] += syn1_delta[c + node_idx];
				}
			}

			// for (c = 0; c < layer1_size; c++){
			// 	syn0_delta[c + index_A] = 0.9 * syn0_delta[c + index_A] - 0.0001 * alpha * syn0[c + index_A] + neu1e[c];
			// 	syn0[c + index_A] += syn0_delta[c + index_A];
			// }

			//<B, B>
			for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_B].codelen; d++) {
				f = 0;
				node_idx = vocab[word_B].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_B] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				}

				// 'g' is the gradient multiplied by the learning rate
				g = (1 - vocab[word_B].code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < layer1_size; c++)
					neu1e[c] += g * syn1[c + node_idx];

				// Learn weights hidden -> output
				for(c=0; c<layer1_size; c++){
					syn1_delta[c + node_idx] =  MOMENTUM * syn1_delta[c + node_idx] - DECAY * alpha * syn1[c + node_idx] + g * syn0[c + index_B];
					syn1[c + node_idx] += syn1_delta[c + node_idx];
				}
			}

			// for (c = 0; c < layer1_size; c++){
			// 	syn0_delta[c + index_B] = 0.9 * syn0_delta[c + index_B] - 0.0001 * alpha * syn0[c + index_B] + neu1e[c];
			// 	syn0[c + index_B] += syn0_delta[c + index_B];
			// }
		// }
		pair_stamp++;
		if(pair_stamp > MAX_PAIRS_NUM) break;
	}
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}
void * FixedW_compute_epoch_loss (void *threadarg){
	long long c, d, word_A, word_B, index_A, index_B, node_idx;
	long long pair_count = 0;
	float f;

	pair <char[MAX_STRING] , char[MAX_STRING]> image_user_pair; //Define a image-user pair.
	long long pair_stamp = 0;

	float *loss_thread;
	float *aa_loss_thread;
	float *ab_loss_thread;
	float *ba_loss_thread;
	float *bb_loss_thread;
	int thread_id;

	struct thread_loss *thread_data;
	thread_data = (struct thread_loss *) threadarg;
	loss_thread = thread_data ->loss_thread;
	aa_loss_thread = thread_data ->aa_loss_thread;
	ab_loss_thread = thread_data ->ab_loss_thread;
	ba_loss_thread = thread_data ->ba_loss_thread;
	bb_loss_thread = thread_data ->bb_loss_thread;
	thread_id = thread_data ->thread_id;

	(*loss_thread) = 0; (*aa_loss_thread) = 0; (*ab_loss_thread) = 0; (*ba_loss_thread) = 0; (*bb_loss_thread) = 0;

	pair_stamp = pair_size / (long long) num_threads  * thread_id;
	string ends = "jpg";
	while(1){
		strncpy(image_user_pair.first, pairs[pair_stamp].first, MAX_STRING);
		strncpy(image_user_pair.second, pairs[pair_stamp].second, MAX_STRING);
		if (pair_stamp >  pair_size) break;
		if (pair_count >  pair_size/ num_threads) break;
		pair_count ++;

		//index of A's in vocabulary
		word_A = SearchVocab(image_user_pair.first);
		if(word_A == -1)
			break;
		//index of B's in vocabulary
		word_B = SearchVocab(image_user_pair.second);
		if(word_B == -1)
			break;

		index_A = word_A * layer1_size;
		index_B = word_B * layer1_size;


		//<A, B>
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word_B].codelen; d++) {
			f = 0;
			node_idx = vocab[word_B].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + index_A] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if(vocab[word_B].code[d] == 0){
					(*loss_thread) += -log(f);
					(*ab_loss_thread) += -log(f);
				}
				else{
					(*loss_thread) += -log(1- f);
					(*ab_loss_thread) += -log(1-f);
				}
			}
		}

		// if (!endsWith (image_user_pair.second, ends.c_str())){
			//<B, A>
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_A].codelen; d++) {
				f = 0;
				node_idx = vocab[word_A].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_B] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					if(vocab[word_A].code[d] == 0){
						(*loss_thread) += -log(f);
						(*ba_loss_thread) += -log(f);
					}
					else{
						(*loss_thread) += -log(1 - f);
						(*ba_loss_thread) += -log(1 - f);
					}
				}
			}

			//<A, A>
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_A].codelen; d++) {
				f = 0;
				node_idx = vocab[word_A].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_A] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					if(vocab[word_A].code[d] == 0){
						(*loss_thread) += -log(f);
						(*aa_loss_thread) += -log(f);
					}
					else{
						(*loss_thread) += -log(1-f);
						(*aa_loss_thread) += -log(1-f);
					}
				}
			}

			//<B, B>
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_B].codelen; d++) {
				f = 0;
				node_idx = vocab[word_B].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += syn0[c + index_B] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else{
					f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					if(vocab[word_B].code[d] == 0){
						(*loss_thread) += -log(f);
						(*bb_loss_thread) += -log(f);
					}
					else{
						(*loss_thread) += - log(1 - f);
						(*bb_loss_thread) += - log(1 - f);
					}
				}
			}
		// }
		pair_stamp++;
		if(pair_stamp > MAX_PAIRS_NUM) break;
	}
	pthread_exit(NULL);
}

void InitNet() {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	if (syn0_init_file[0] && strstr(syn0_init_file, ".txt")) {
		int n, m, idx;
		char word[MAX_STRING];
		FILE *fp;
		fp = fopen(syn0_init_file, "r");
		printf("Loading initial syn0 from ascii file...\n");
		fscanf(fp, "%d %d\n", &n, &m);
		assert(n == vocab_size && m == layer1_size);
		for (n = 0; n < vocab_size; ++n) {
			fscanf(fp, "%s", word);
			idx = SearchVocab(word);
			assert(idx != -1);
			for (m = 0; m < layer1_size; ++m) {
				fscanf(fp, "%f ", &(syn0[idx * layer1_size + m]));
			}
		}
		fclose(fp);
	}
  else if (syn0_init_file[0] && strstr(syn0_init_file, ".h5")) {
    printf("Loading initial syn0 from hdf5 file...\n");
    hid_t file_id = H5Fopen(syn0_init_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file_id >= 0);
    herr_t status = H5LTread_dataset_float(file_id, "/feat", syn0);
    assert(status >= 0);
    status = H5Fclose(file_id);
    assert(status >= 0);
  }
	else {
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;  // For each vocabulary word, it corresponds to a hidden value.
	}

	a = posix_memalign((void **)&syn0_best, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_best[a * layer1_size + b] = syn0[a * layer1_size + b];

	a = posix_memalign((void **)&syn0_delta, 128, (long long)vocab_size * layer1_size * sizeof(real));  //record the old value of the parameters.
	if (syn0_delta == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_delta[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn1_delta, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn1_delta == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_delta[a * layer1_size + b] = 0.0;

	CreateBinaryTree();
}

void TrainModel() {  //here different threads share variables. so they can implement the parallel processing.
	//Train the models first, get the word vectors, then save the word vectors or cluster the word vectors.
	long a, b;
	FILE *fo_best, *f1_best, *f1_curr;
	float epoch_loss = 0, aa_epoch_loss = 0, ab_epoch_loss = 0, ba_epoch_loss = 0, bb_epoch_loss = 0;

	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0) 
    ReadVocab(); 
  else 
    LearnVocabFromTrainFile();
	if (save_vocab_file[0] != 0) 
    SaveVocab();
	LearnPairsFromFile ();
	InitNet();
	start = clock();
	double epoch_loss_best = 1000;
	cout << "pair_size: \t " << pair_size << "\t total_size:\t " << total_pairs << endl;

	//FindMaxDegreeMultiply();

	float *loss_thread = new float[num_threads];
	float *aa_loss_thread = new float[num_threads];
	float *bb_loss_thread = new float[num_threads];
	float *ab_loss_thread = new float[num_threads];
	float *ba_loss_thread = new float[num_threads];
	struct thread_loss loss_struct[num_threads];
	cout << "Epoch No: -1 " <<endl;
	for(a = 0; a < num_threads; a++){
		loss_struct[a].loss_thread = &loss_thread[a];
		loss_struct[a].aa_loss_thread = &aa_loss_thread[a];
		loss_struct[a].ab_loss_thread = &ab_loss_thread[a];
		loss_struct[a].ba_loss_thread = &ba_loss_thread[a];
		loss_struct[a].bb_loss_thread = &bb_loss_thread[a];
		loss_struct[a].thread_id = a;
		pthread_create(&pt[a], NULL, FixedW_compute_epoch_loss, (void *) &loss_struct[a]);
	}
	for (a = 0; a < num_threads; a ++) pthread_join(pt[a], NULL);

	for (int i = 0; i < num_threads; i++){
		epoch_loss += loss_thread[i];
		aa_epoch_loss += aa_loss_thread[i];
		ab_epoch_loss += ab_loss_thread[i];
		ba_epoch_loss += ba_loss_thread[i];
		bb_epoch_loss += bb_loss_thread[i];
	}

	epoch_loss = epoch_loss/ total_pairs;
	aa_epoch_loss = aa_epoch_loss/total_pairs;
	ab_epoch_loss = ab_epoch_loss/total_pairs;
	ba_epoch_loss = ba_epoch_loss/total_pairs;
	bb_epoch_loss = bb_epoch_loss/total_pairs;

	printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", epoch_loss, aa_epoch_loss, ab_epoch_loss, ba_epoch_loss, bb_epoch_loss);
	epoch_loss = 0; aa_epoch_loss = 0; ab_epoch_loss = 0; ba_epoch_loss = 0; bb_epoch_loss = 0;
	for (int e = 0; e < num_epoches; e ++)
	{
		alpha = starting_alpha * (1 - e / (real)(num_epoches + 1));
		if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;

		cout << "epoch no:  " <<e<<endl;

		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); //Change alpha only in the first threads;
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);  // wait for all the sub threads finished.

		//train model on negative samples TrainModelNegThread
//		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelNegThread, (void *)a);
//		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);  // wait for all the sub threads finished.

		//Compute loss before update.
		for(a = 0; a < num_threads; a++){
			loss_struct[a].loss_thread = &loss_thread[a];
			loss_struct[a].aa_loss_thread = &aa_loss_thread[a];
			loss_struct[a].ab_loss_thread = &ab_loss_thread[a];
			loss_struct[a].ba_loss_thread = &ba_loss_thread[a];
			loss_struct[a].bb_loss_thread = &bb_loss_thread[a];
			loss_struct[a].thread_id = a;
			pthread_create(&pt[a], NULL, FixedW_compute_epoch_loss, (void *) &loss_struct[a]);
		}
		for (a = 0; a < num_threads; a ++) pthread_join(pt[a], NULL);

		for (int i = 0; i < num_threads; i++){
			epoch_loss += loss_thread[i];
			aa_epoch_loss += aa_loss_thread[i];
			ab_epoch_loss += ab_loss_thread[i];
			ba_epoch_loss += ba_loss_thread[i];
			bb_epoch_loss += bb_loss_thread[i];
		}

		epoch_loss = epoch_loss/ total_pairs;
		aa_epoch_loss = aa_epoch_loss/total_pairs;
		ab_epoch_loss = ab_epoch_loss/total_pairs;
		ba_epoch_loss = ba_epoch_loss/total_pairs;
		bb_epoch_loss = bb_epoch_loss/total_pairs;

		printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", epoch_loss, aa_epoch_loss,
				ab_epoch_loss, ba_epoch_loss, bb_epoch_loss);

    if (e % 100 == 0) {
      f1_curr = fopen(syn1_curr_file, "wb");
      fprintf(f1_curr, "%lf %d\n", epoch_loss, e);
      for (a = 0; a < vocab_size; a++) {
        fprintf(f1_curr, "%s ", vocab[a].word);
        for (b = 0; b < layer1_size; b++)   
          fprintf(f1_curr, "%lf ", syn1[a * layer1_size + b]);
        fprintf(f1_curr, "\n");
      }
      fclose(f1_curr);
    }
		if(epoch_loss_best > epoch_loss){
  		epoch_loss_best = epoch_loss;

			// Save the word vectors
			/*for (a = 0; a < vocab_size; a++)
				for (b = 0; b < layer1_size; b++)
					syn0_best[a * layer1_size + b] = syn0[a * layer1_size + b];

			fo_best = fopen(syn0_best_file, "wb");
			fprintf(fo_best, "%lf %lld %lld\n", epoch_loss_best, vocab_size, layer1_size);
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo_best, "%s ", vocab[a].word);
				if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0_best[a * layer1_size + b], sizeof(real), 1, fo_best);
				else for (b = 0; b < layer1_size; b++) 	fprintf(fo_best, "%lf ", syn0_best[a * layer1_size + b]);
				fprintf(fo_best, "\n");
			}
			fclose(fo_best);*/

			f1_best = fopen(syn1_best_file, "wb");
			fprintf(f1_best, "%lf %lld %lld\n", epoch_loss_best, vocab_size, layer1_size);
			for (a = 0; a < vocab_size; a++) {
				fprintf(f1_best, "%s ", vocab[a].word);
				//				cout << "syn1[a * layer1_size]:    " <<syn1[a * layer1_size] <<endl;
				for (b = 0; b < layer1_size; b++) 	fprintf(f1_best, "%lf ", syn1[a * layer1_size + b]);
				fprintf(f1_best, "\n");
			}
			fclose(f1_best);

			//			fo_syn0_delta = fopen(syn0_delta_file, "wb");
			//			fprintf(fo_syn0_delta, "%f %f %lld %lld\n", alpha, epoch_loss_best, vocab_size, layer1_size);
			//			for (a = 0; a < vocab_size; a++) {
			//				fprintf(fo_syn0_delta, "%s ", vocab[a].word);
			//				//				cout << "syn1[a * layer1_size]:    " <<syn1[a * layer1_size] <<endl;
			//				for (b = 0; b < layer1_size; b++) 	fprintf(fo_syn0_delta, "%lf ", syn0_delta[a * layer1_size + b]);
			//				fprintf(fo_syn0_delta, "\n");
			//			}
			//			fclose(fo_syn0_delta);
			//
			//			fo_syn1_delta = fopen(syn1_delta_file, "wb");
			//			fprintf(fo_syn1_delta, "%f %f %lld %lld\n", alpha, epoch_loss_best, vocab_size, layer1_size);
			//			for (a = 0; a < vocab_size; a++) {
			//				fprintf(fo_syn1_delta, "%s ", vocab[a].word);
			//				//				cout << "syn1[a * layer1_size]:    " <<syn1[a * layer1_size] <<endl;
			//				for (b = 0; b < layer1_size; b++) 	fprintf(fo_syn1_delta, "%lf ", syn1_delta[a * layer1_size + b]);
			//				fprintf(fo_syn1_delta, "\n");
			//			}
			//			fclose(fo_syn1_delta);

		}
		epoch_loss = 0; aa_epoch_loss = 0; ab_epoch_loss = 0; ba_epoch_loss = 0; bb_epoch_loss = 0;
	}
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	syn0_best_file[0] = 0;


	//	layer1_size = 300;
	//	strncpy(train_file, "./Data/SampleData.txt", MAX_STRING);
	//	//	strncpy (featFile, "./Data/SampleFeature.txt", MAX_STRING);
	//	//	strncpy (imageIndexFile, "./Data/SampleImageIndex.txt", MAX_STRING);
	//	strncpy (output_file, "./Result/1015Vectors.txt", MAX_STRING);
	//	strncpy (syn0_best_file, "./Result/1015Syn0Result.txt", MAX_STRING);
	//	strncpy (save_vocab_file, "./Result/1015vocab.txt", MAX_STRING);
	//	//	strncpy (w_best_file, "./Result/1015WBest.txt", MAX_STRING);
	//	num_threads = 1;
	//	num_epoches = 10000000;
	//	min_count = 1;

	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) num_epoches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-syn0-best", argc, argv)) > 0) strcpy(syn0_best_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-syn1-best", argc, argv)) > 0) strcpy(syn1_best_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn1-curr", argc, argv)) > 0) strcpy(syn1_curr_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tree-file", argc, argv)) > 0) strcpy(tree_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-syn0-init", argc, argv)) > 0)	strcpy(syn0_init_file, argv[i + 1]);
	//	if ((i = ArgPos((char *)"-syn0-delta", argc, argv)) > 0) strcpy(syn0_delta_file, argv[i + 1]);
	//	if ((i = ArgPos((char *)"-syn1-delta", argc, argv)) > 0) strcpy(syn1_delta_file, argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  pairs = new pair<char[MAX_STRING], char[MAX_STRING]>[MAX_PAIRS_NUM];
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE + 1; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}
