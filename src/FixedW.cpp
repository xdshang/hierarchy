/*
 * FixedW.cpp
 *
 *  Created on: Oct 28, 2014
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
#include <algorithm>
#include <set>
#include <map>
using namespace std;

#include "FixedW.h"
#include "ValueDef.h"
#include "FixedSy.h"
#include "Word2Vec.h"
#include "CommonFunction.h"

int read_w = 0, read_sy_delta = 0;
float fixedW_alpha, fixedW_starting_alpha;
char fixed_w_loss_file[MAX_STRING];

void *FixedW_TrainModelThread(void *id) {
	long long d, word, last_word;
	long long pair_count = 0;
	long long l1, l2, c;
	float f, g;
	float *neu1 = (float *)calloc(layer1_size, sizeof(float));
	float *neu1e = (float *)calloc(layer1_size, sizeof(float));

	//	cout << "FixedW_alpha:" << fixedW_alpha <<endl;
	pair <char[MAX_STRING] , char[MAX_STRING]> image_user_pair; //Define a image-user pair.
	int pair_stamp = 0;
	pair_stamp = pair_size / (long long)num_threads  * (long long)id;
	while(1){
		strncpy(image_user_pair.first, pairs[pair_stamp].first, MAX_STRING);
		strncpy(image_user_pair.second, pairs[pair_stamp].second, MAX_STRING);
		if (pair_count >  pair_size/ num_threads) break;
		pair_count ++;

		//<A, B>
		last_word = SearchVocab(image_user_pair.first);
		if(last_word == -1)
			break;
		word = SearchVocab(image_user_pair.second);
		if(last_word == -1)
			break;

		l1 = last_word * layer1_size;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			}

			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word].code[d] - f) * fixedW_alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++)
				neu1e[c] += g * syn1[c + l2];

			// Learn weights hidden -> output
			for(c=0; c<layer1_size; c++){
				syn1_delta[c+l2] =  0.9 * syn1_delta[c+l2]-0.0001*fixedW_alpha*syn1[c+l2] + g * syn0[c + l1];
				syn1[c + l2] += syn1_delta[c+l2];
			}
		}

		//Compute mini batch loss and add to the neu1e.
		for (c = 0; c < layer1_size; c++)
		{
			syn0_delta[c + l1] = 0.9 * syn0_delta[c + l1] - 0.0001 * fixedW_alpha * syn0[c + l1] + neu1e[c];
			syn0[c + l1] += syn0_delta[c + l1];
		}

		//<B, A>
		last_word = SearchVocab(image_user_pair.second);
		if(last_word == -1) break;
		word = SearchVocab(image_user_pair.first);

		l1 = last_word * layer1_size;

		for (c = 0; c < layer1_size; c++)
			neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierachical node and changed. Here l2 is for the specific word.
			//			cout << "pair probability:  "<< pair_probablity <<endl;

			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else
			{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			}
			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word].code[d] - f) * fixedW_alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++)
				neu1e[c] += g * syn1[c + l2];

			// No Update Image Node.
			for(c=0; c<layer1_size; c++){
				syn1_delta[c+l2] =  0.9 * syn1_delta[c+l2] - 0.0001 * fixedW_alpha * syn1[c+l2] + g * syn0[c + l1];
				syn1[c + l2] += syn1_delta[c+l2];
			}
		}

		for (c = 0; c < layer1_size; c++){
			syn0_delta[c + l1] = 0.9 * syn0_delta[c + l1] - 0.0001 * fixedW_alpha * syn0[c + l1] + neu1e[c];
			syn0[c + l1] += syn0_delta[c + l1];
		}

		//<A, A>
		last_word = SearchVocab(image_user_pair.first);
		if(last_word == -1)
			break;
		word = SearchVocab(image_user_pair.first);
		if(last_word == -1)
			break;

		l1 = last_word * layer1_size;
		//		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word].code[d] - f) * fixedW_alpha;
			// Learn weights hidden -> output
			//No update Image nodes.
			for(c=0; c<layer1_size; c++){
				syn1_delta[c+l2] =  0.9 * syn1_delta[c+l2]-0.0001*fixedW_alpha*syn1[c+l2] + g * syn0[c + l1];
				syn1[c + l2] += syn1_delta[c+l2];
			}
		}

		//Compute mini batch loss and add to the neu1e.
		for (c = 0; c < layer1_size; c++)
		{
			syn0_delta[c + l1] = 0.9 * syn0_delta[c + l1] - 0.0001 * fixedW_alpha * syn0[c + l1] + neu1e[c];
			syn0[c + l1] += syn0_delta[c + l1];
		}

		//<B, B>
		last_word = SearchVocab(image_user_pair.second);
		if(last_word == -1)
			break;
		word = SearchVocab(image_user_pair.second);
		if(last_word == -1)
			break;

		l1 = last_word * layer1_size;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else
			{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			}

			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word].code[d] - f) * fixedW_alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++)
				neu1e[c] += g * syn1[c + l2];

			// Learn weights hidden -> output
			for(c=0; c<layer1_size; c++){
				syn1_delta[c+l2] =  0.9 * syn1_delta[c+l2]-0.0001*fixedW_alpha*syn1[c+l2] + g * syn0[c + l1];
				syn1[c + l2] += syn1_delta[c+l2];
			}
		}

		for (c = 0; c < layer1_size; c++){
			syn0_delta[c + l1] = 0.9 * syn0_delta[c + l1] - 0.0001 * fixedW_alpha * syn0[c + l1] + neu1e[c];
			syn0[c + l1] += syn0_delta[c + l1];
		}

		pair_stamp ++;
		if(pair_stamp > MAX_PAIRS_NUM) break;
	}
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void * FixedW_compute_epoch_loss (void *threadarg){
	long long d, word, last_word;
	long long pair_count = 0;
	long long l1, l2, c;
	float f;

	pair <char[MAX_STRING] , char[MAX_STRING]> image_user_pair; //Define a image-user pair.
	int pair_stamp = 0;

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
	while(1){
		//		image_user_pair = pairs[pair_stamp];
		strncpy(image_user_pair.first, pairs[pair_stamp].first, MAX_STRING);
		strncpy(image_user_pair.second, pairs[pair_stamp].second, MAX_STRING);
		//		cout<<"image user pairs  "<<pair_stamp <<"   "<<image_user_pair.first<<"   "<<image_user_pair.second<<endl;
		if (pair_stamp >  pair_size) break;
		if (pair_count >  pair_size/ num_threads) break;
		pair_count ++;

		//<A, B>
		last_word = SearchVocab(image_user_pair.first);
		if(last_word == -1) {
			break;
		}
		word = SearchVocab(image_user_pair.second);
		if(last_word == -1) {
			break;
		}
		l1 = last_word * layer1_size;
		//		double pair_loss = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else
			{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if(vocab[word].code[d] == 0){
					(*loss_thread) += -log(f);
					(*ab_loss_thread) += -log(f);
				}
				else{
					(*loss_thread) += -log(1- f);
					(*ab_loss_thread) += -log(1-f);
				}
			}
		}

		//<B, A>
		last_word = SearchVocab(image_user_pair.second);
		word = SearchVocab(image_user_pair.first);
		l1 = last_word * layer1_size;

		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else
			{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if(vocab[word].code[d] == 0){
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
		last_word = SearchVocab(image_user_pair.first);
		word = SearchVocab(image_user_pair.first);
		l1 = last_word * layer1_size;

		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if(vocab[word].code[d] == 0){
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
		last_word = SearchVocab(image_user_pair.second);
		word = SearchVocab(image_user_pair.second);
		l1 = last_word * layer1_size;

		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else
			{
				f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				if(vocab[word].code[d] == 0){
					(*loss_thread) += -log(f);
					(*bb_loss_thread) += -log(f);
				}
				else{
					(*loss_thread) += - log(1 - f);
					(*bb_loss_thread) += - log(1 - f);
				}
			}
		}
		pair_stamp++;
		if(pair_stamp > MAX_PAIRS_NUM) break;
	}
	pthread_exit(NULL);
}

void FixedW_InitNet() {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

	a = posix_memalign((void **)&syn0_delta, 128, (long long)vocab_size * layer1_size * sizeof(float));  //record the old value of the parameters.
	if (syn0_delta == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_delta[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1[a * layer1_size + b] = 0;

	a = posix_memalign((void **)&syn1_delta, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn1_delta == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_delta[a * layer1_size + b] = 0.0;

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (float)RAND_MAX - 0.5) / layer1_size;  // For each vocabulary word, it corresponds to a hidden value.

	CreateBinaryTree();
}
/*
 * read w from file.
 */
void ReadParaW(){ //Has been tested.
	char word[MAX_STRING];
	FILE *fin;
	int rows = 0, cols = 0;
	//Read W
	if(read_w == 1){
		fin = fopen(w_best_file, "rb");
		if (fin == NULL) {
			printf("ERROR: W file not found!\n");
			exit(1);
		}
		while (1) {
			ReadWord(word, fin);
			if (feof(fin)) break;  // file end, break out.
			if(strcmp(word, (char *)"</s>") == 0)
			{
				rows ++;
				cols = 0;
				continue;
			}
			tranMatrix[rows * feat_size + cols] = std::stof(word);
			//			if (cols == 1) cout << "tranMatrix[rows * feat_size + cols]" << rows << "   " << cols <<  "  " << tranMatrix[rows * feat_size + cols] << endl;
			cols ++;
		}
		fclose(fin);
		rows = 0; cols = 0;
	}
}
/** Initialize syn0 of images using w.
 */
void GetImSyn0(){
	int index = 0, c, p, t;
	string ends = "jpg";
	for (int i = 0; i < vocab_size; i++) {
		if (endsWith (vocab[i].word, ends.c_str())){
			index = image_index[vocab[i].word]; //index starts from 0;
			for(c = 0; c < layer1_size; c ++){
				syn0[i * layer1_size + c] = 0;
				for (t = 0; t < nonzeroLen[index]; t ++){
					p = Idx[nonzeroIdx[index] + t];
					syn0[i * layer1_size + c] += tranMatrix[ c * feat_size + p - 1] * Val[nonzeroIdx[index] + t];
				}
			}
		}
	}
}

void FixedW_ResetPara()
{
	int a, b;
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_delta[a * layer1_size + b] = 0.0;

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_delta[a * layer1_size + b] = 0.0;
}

void FixedW_TrainModel() {  //here different threads share variables. so they can implement the parallel processing.
	//Train the models first, get the word vectors, then save the word vectors or cluster the word vectors.
	long a, b;
	FILE *fo_loss, *f1;
	float epoch_loss = 0, aa_epoch_loss = 0, ab_epoch_loss = 0, ba_epoch_loss = 0, bb_epoch_loss = 0, last_epoch_loss = 0;
	cout << "---------------------------------------------------------------------" <<endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);

//	ReadParaW(); //Tested
	GetImSyn0(); //Tested

	printf("Fixed Sy Learning Rate: %f\t Scale %d\tPair_size %lld \t \n", fixedW_starting_alpha, scale, pair_size);
	//Save Syn0 -> tested loading
	//	fo_best = fopen(syn0_best_file, "wb");
	//	fprintf(fo_best, "%lf %lld %d\n", epoch_loss_best, vocab_size, layer1_size);
	//	for (a = 0; a < vocab_size; a++) {
	//		fprintf(fo_best, "%s ", vocab[a].word);
	//		if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo_best);
	//		else for (b = 0; b < layer1_size; b++) 	fprintf(fo_best, "%lf ", syn0[a * layer1_size + b]);
	//		fprintf(fo_best, "\n");
	//	}
	//	fclose(fo_best);

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

	long long temp = 4 * pair_size;
	epoch_loss = epoch_loss/ temp;
	aa_epoch_loss = aa_epoch_loss/temp;
	ab_epoch_loss = ab_epoch_loss/temp;
	ba_epoch_loss = ba_epoch_loss/temp;
	bb_epoch_loss = bb_epoch_loss/temp;

	printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", epoch_loss, aa_epoch_loss, ab_epoch_loss, ba_epoch_loss, bb_epoch_loss);
	fo_loss = fopen(fixed_w_loss_file, "ab");
	fprintf(fo_loss, "Epoch No: %d, Loss: %lf\n", -1, epoch_loss);
	fclose(fo_loss);
	last_epoch_loss = epoch_loss;
	epoch_loss = 0; aa_epoch_loss = 0; ab_epoch_loss = 0; ba_epoch_loss = 0; bb_epoch_loss = 0;
	int e = 0;
	for (e = 0; e < num_epoches; e++)
	{
		fixedW_alpha = fixedW_starting_alpha * (1 - e / (float)(num_epoches + 1));
		if (fixedW_alpha < fixedW_starting_alpha * 0.001) fixedW_alpha = fixedW_starting_alpha * 0.001;

		cout << "epoch no:  " <<e<<endl;

		//Update
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, FixedW_TrainModelThread, (void *)a); //Change fixedW_alpha only in the first threads;
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);  // wait for all the sub threads finished.

		//Compute loss
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

		long long temp = 4 * pair_size;
		epoch_loss = epoch_loss/ temp;
		aa_epoch_loss = aa_epoch_loss/temp;
		ab_epoch_loss = ab_epoch_loss/temp;
		ba_epoch_loss = ba_epoch_loss/temp;
		bb_epoch_loss = bb_epoch_loss/temp;

		printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", epoch_loss, aa_epoch_loss, ab_epoch_loss, ba_epoch_loss, bb_epoch_loss);
		fo_loss = fopen(fixed_w_loss_file, "ab");
		fprintf(fo_loss, "Epoch No: %d, Loss: %lf\n", e, epoch_loss);
		fclose(fo_loss);

		if(iter_loss > epoch_loss){
			iter_loss = epoch_loss;

			f1 = fopen(syn0_temp_file, "wb");
			fixedw_alpha_best = fixedW_alpha;
			for (a = 0; a < vocab_size; a++) {
				for (b= 0; b < layer1_size; b++){
					syn0_best[a * layer1_size + b] = syn0[a * layer1_size + b];
					syn1_best[a * layer1_size + b] = syn1[a * layer1_size + b];

					fprintf(f1, "%lf ", syn0_best[a * layer1_size + b]);
				}
				fprintf(f1, "\n");
			}
			fclose(f1);
		}

		if (abs(last_epoch_loss - epoch_loss) /epoch_loss < 0.001)
		{
			free(loss_thread);
			free(aa_loss_thread);
			free(ab_loss_thread);
			free(ba_loss_thread);
			free(bb_loss_thread);
			return;
		}
		last_epoch_loss = epoch_loss;
		epoch_loss = 0; aa_epoch_loss = 0; ab_epoch_loss = 0; ba_epoch_loss = 0; bb_epoch_loss = 0;
	}
	free(loss_thread);
	free(aa_loss_thread);
	free(ab_loss_thread);
	free(ba_loss_thread);
	free(bb_loss_thread);
	return;
}
