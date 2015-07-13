/*
 * ImSoftMaxVector.cpp
 *
 *  Created on: Oct 13, 2014
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
#include <random>
#include <set>
#include <iomanip>
#include <cmath>
#include <map>
#include <time.h>
#include <vector>
#include <algorithm>
using namespace std;

#include "FixedSy.h"
#include "FixedW.h"
#include "ValueDef.h"
#include "Word2Vec.h"
#include "CommonFunction.h"

int read_sy = 0;
float fixedsy_alpha, fixedsy_starting_alpha;
char fixed_sy_loss_file[MAX_STRING];

void *GetEachPairUpdate(void *threadarg){
	long long c, d,word_A, word_B, image_index_A, node_idx;;
	float f, g;
	int k, t;
	int p =0;
	int thread_id = 0;
	int mini_index = 0;
	int mini_pairsno = 0;
	float *neu1e;
	float *syn0_temp;

	struct thread_par *thread_data;
	thread_data = (struct thread_par *) threadarg;
	thread_id = thread_data->thread_id;
	mini_index = thread_data->mini_index;
	mini_pairsno = thread_data->mini_pairsno;
	neu1e = thread_data->neu1e;
	syn0_temp = thread_data->syn0_temp;

	pair <char[MAX_STRING], char[MAX_STRING]> image_user_pair;
	strncpy(image_user_pair.first, pairs[randomVector.at(mini_index * mini_pairsno + thread_id)].first, MAX_STRING); //Compute loss and Save
	//	cout << "image first" << image_user_pair.first <<endl;
	strncpy(image_user_pair.second, pairs[randomVector.at(mini_index * mini_pairsno + thread_id)].second, MAX_STRING); //Compute loss and Save

	string ends = "jpg";
	//index of A's in vocabulary
	word_A = SearchVocab(image_user_pair.first);
	if(word_A == -1){
    cout << image_user_pair.first << "not found in vocab." << endl;
		pthread_exit(NULL);
  }
	//index of B's in vocabulary
	word_B = SearchVocab(image_user_pair.second);
	if(word_B == -1){
    cout << image_user_pair.second << "not found in vocab." << endl;
		pthread_exit(NULL);
  }

	//index of A in the syn0
	image_index_A = image_index.at(image_user_pair.first);

	//<A, B>
	for (c = 0; c < layer1_size; c++){
		neu1e[c] = 0;
		syn0_temp[c] = 0;
	}

	for(c = 0; c < layer1_size; c ++)
		for (t = 0; t < nonzeroLen[image_index_A]; t ++){
			p = Idx[nonzeroIdx[image_index_A] + t];
			syn0_temp[c] += tranMatrix[ c * feat_size + p - 1] * Val[nonzeroIdx[image_index_A] + t];
		}

	// HIERARCHICAL SOFTMAX
	for (d = 0; d < vocab[word_B].codelen; d++) {
		f = 0;
		node_idx = vocab[word_B].point[d] * layer1_size;

		// Propagate hidden -> output
		for (c = 0; c < layer1_size; c++) f += syn0_temp[c] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
		f = 1/(1+exp(-f));
		// 'g' is the gradient multiplied by the learning rate
		g = (1 - vocab[word_B].code[d] - f) * fixedsy_alpha;
		// Propagate errors output -> hidden
		for (c = 0; c < layer1_size; c++)
			neu1e[c] += g * syn1[c + node_idx];
	}
	//updated tranMatrix
	for (k = 0; k < layer1_size; k++)
		for (t = 0; t < nonzeroLen[image_index_A]; t++){
			p = Idx[nonzeroIdx[image_index_A] + t]; // 1 - 4096.
			tranMatrix_update[k * feat_size + p - 1] += neu1e[k] * Val[nonzeroIdx[image_index_A] + t];
		}

	/*if(!endsWith (image_user_pair.second, ends.c_str())){
		//<A, A>
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word_A].codelen; d++) {
			f = 0;
			node_idx = vocab[word_A].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += syn0_temp[c] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
			f = 1/(1+exp(-f));
			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word_A].code[d] - f) * fixedsy_alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++)
				neu1e[c] += g * syn1[c + node_idx];
		}

		for (k = 0; k <layer1_size; k ++)
			for (t = 0; t < nonzeroLen[image_index_A]; t++){
				p = Idx[nonzeroIdx[image_index_A] + t];
				tranMatrix_update[k * feat_size + p - 1] += neu1e[k] * Val[nonzeroIdx[image_index_A] + t];
			}
	}*/
	pthread_exit(NULL);
}

void UpdateEpoch()
{
	int c, r;
	float norm;
	cout << "Pair_size:  " <<pair_size<<endl;
  cout << "Batch_size: " << batch_size << endl;
	for(long long i = 0; i < pair_size; i++)
		randomVector.push_back(i);
	std::random_shuffle(randomVector.begin(), randomVector.end());
	int mini_pairsno = batch_size/4;
	int no_batches = ceil(pair_size/mini_pairsno);

	float **neu1e = new float*[mini_pairsno];
	for (c = 0; c < mini_pairsno; c++)
		neu1e[c] = new float[layer1_size];
	float **syn0_temp = new float*[mini_pairsno];
	for (c = 0; c < mini_pairsno; c++)
		syn0_temp[c] = new float[layer1_size];

	cout << "Number Batches ****************** :" << no_batches <<endl;
	pthread_t *pt = (pthread_t *)malloc(mini_pairsno * sizeof(pthread_t)); //allocate threads addresses.
	for(int j = 0; j < no_batches; j++){
		struct thread_par td[mini_pairsno];
		for(c = 0; c < mini_pairsno; c++){
			td[c].thread_id = c;
			td[c].mini_index = j;
			td[c].mini_pairsno = mini_pairsno;
			td[c].neu1e = neu1e[c];
			td[c].syn0_temp = syn0_temp[c];
			pthread_create(&pt[c], NULL, GetEachPairUpdate, (void *) &td[c]);
		}
		for (c = 0; c < mini_pairsno; c ++) pthread_join(pt[c], NULL);
		if(j % 1000 == 0)
			cout << "Minibatch_no:   " << j << endl;

		//hanwang, mini-batch update
		//update W
		for (r = 0; r < layer1_size; r++){
			norm = 0;
			for (c = 0; c < feat_size; c++){
				tranMatrix_delta[r*feat_size+c] = 0.9 * tranMatrix_delta[r*feat_size+c] -  0*fixedsy_alpha * tranMatrix[r*feat_size+c] + tranMatrix_update[r*feat_size+c]/(batch_size/2);
				tranMatrix[r*feat_size+c] +=  tranMatrix_delta[r*feat_size+c];
				tranMatrix_update[r*feat_size+c] = 0;
				norm += tranMatrix[r*feat_size+c]*tranMatrix[r*feat_size+c];
			}
			norm = sqrt(norm);
			if (norm > max_norm)
				for(c = 0; c < feat_size; c ++)
					tranMatrix[r * feat_size + c] = max_norm * tranMatrix[r * feat_size + c]/norm;
		}
	}
	randomVector.erase(randomVector.begin(), randomVector.end());
	delete[] neu1e;
	delete[] syn0_temp;
}

void * FixedSy_compute_epoch_loss (void *threadarg){
	long long c, d, word_A, word_B, image_index_A, node_idx;
	long long pair_count = 0;
	float f;
	int p=0, t=0, pair_stamp = 0;
	float *temp = (float *)calloc(layer1_size, sizeof(float));
	pair <char[MAX_STRING] , char[MAX_STRING]> image_user_pair; //Define a image-user pair.

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
		strncpy(image_user_pair.first, pairs[pair_stamp].first, MAX_STRING);
		strncpy(image_user_pair.second, pairs[pair_stamp].second, MAX_STRING);

		string ends = "jpg";
		//index of A's in vocabulary
		word_A = SearchVocab(image_user_pair.first);
		if(word_A == -1)
			pthread_exit(NULL);
		//index of B's in vocabulary
		word_B = SearchVocab(image_user_pair.second);
		if(word_B == -1)
			pthread_exit(NULL);

		//index of A in the syn0
		image_index_A =  image_index[image_user_pair.first];

		if (pair_stamp >  pair_size) break;
		if (pair_count >  pair_size/ num_threads) break;
		pair_count ++;

		//<A, B>
		for (c = 0; c < layer1_size; c++) temp[c] = 0;
		for(c = 0; c < layer1_size; c ++)
			for (t = 0; t < nonzeroLen[image_index_A]; t ++){
				p = Idx[nonzeroIdx[image_index_A] + t];
				temp[c] += tranMatrix[ c * feat_size + p - 1] * Val[nonzeroIdx[image_index_A] + t];
			}

		// HIERARCHICAL SOFTMAX
		for (d = 0; d < vocab[word_B].codelen; d++) {
			f = 0;
			node_idx = vocab[word_B].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++)
				f += temp[c] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.

			//			f = 1/(1+exp(-f));
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

			if(vocab[word_B].code[d] == 0){
				(*ab_loss_thread) += - log(f);
				(*loss_thread) += - log(f);
			}
			else{
				(*loss_thread) += - log(1-f);
				(*ab_loss_thread) +=  - log(1-f);
			}
		}

		//<A, A>
		if(! endsWith (image_user_pair.second, ends.c_str())){
			image_index_A = image_index[image_user_pair.first];
			for (c = 0; c < layer1_size; c++) temp[c] = 0;
			for(c = 0; c < layer1_size; c ++)
				for (t = 0; t < nonzeroLen[image_index_A]; t ++){
					p = Idx[nonzeroIdx[image_index_A] + t];
					temp[c] += tranMatrix[ c * feat_size + p - 1] * Val[nonzeroIdx[image_index_A] + t];
				}
			// HIERARCHICAL SOFTMAX
			for (d = 0; d < vocab[word_A].codelen; d++) {
				f = 0;
				node_idx = vocab[word_A].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += temp[c] * syn1[c + node_idx];  //l1 is the near point and keep the same. l2 is the hierarchical node and changed. Here l2 is for the specific word.
				//			f = 1/(1+exp(-f));
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

				if(vocab[word_A].code[d] == 0){
					(*loss_thread) += - log(f);
					(*aa_loss_thread) += - log(f);
				}
				else{
					(*loss_thread) += - log(1 - f);
					(*aa_loss_thread) +=  - log(1 - f);
				}
			}
		}

		pair_stamp++;
		if(pair_stamp > MAX_PAIRS_NUM) break;
	}
	free(temp);
	pthread_exit(NULL);
}

void InitNet() {
	long long a, b, p;
	int r,c;
	float norm;
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


	a = posix_memalign((void **)&syn0_best, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn0_best == NULL) {printf("Memory allocation failed\n"); exit(1);}

	a = posix_memalign((void **)&syn0_delta_best, 128, (long long)vocab_size * layer1_size * sizeof(float));  //record the old value of the parameters.
	if (syn0_delta_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_delta_best[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn1_best, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn1_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_best[a * layer1_size + b] = 0;

	a = posix_memalign((void **)&syn1_delta_best, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn1_delta_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_delta_best[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn1_update, 128, (long long)vocab_size * layer1_size * sizeof(float));
	if (syn1_update == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn1_update[a * layer1_size + b] = 0.0;

	a = posix_memalign((void **)&syn0_update, 128, (long long)vocab_size * layer1_size * sizeof(float));  //record the old value of the parameters.
	if (syn0_update == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0_update[a * layer1_size + b] = 0.0;

	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (float)RAND_MAX - 0.5) / layer1_size;  // For each vocabulary word, it corresponds to a hidden value.

	for (a = 0; a < vocab_size; a ++){
		norm = 0;
		p = a * layer1_size;
		for (c = 0; c < layer1_size; c ++){
			norm += syn0[p + c] * syn0[p + c];
		}
		norm = sqrt(norm);
		if (norm > max_norm)
			for (c = 0; c < layer1_size; c ++)
				syn0[p + c] = max_norm * syn0[p + c]/norm;
	}

	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix [a * feat_size + b] = ((double) rand() / (RAND_MAX));

	for (r = 0; r < layer1_size; r ++){
		norm = 0;
		for (c = 0; c <feat_size; c ++)
			norm += tranMatrix[r*feat_size+c] * tranMatrix[r*feat_size+c];
		norm = sqrt(norm);
		if (norm > max_norm)
			for(c = 0; c <feat_size; c ++)
				tranMatrix[r*feat_size+c] = max_norm * tranMatrix[r*feat_size+c]/norm;
	}

	a = posix_memalign((void **)&tranMatrix_delta, 128, (long long)feat_size * layer1_size * sizeof(float));
	if (tranMatrix_delta == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix_delta [a * feat_size + b] = 0;

	a = posix_memalign((void **)&tranMatrix_delta_best, 128, (long long)feat_size * layer1_size * sizeof(float));
	if (tranMatrix_delta_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix_delta_best [a * feat_size + b] = 0;

	a = posix_memalign((void **)&tranMatrix_best, 128, (long long)feat_size * layer1_size * sizeof(float));
	if (tranMatrix_best == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix_best [a * feat_size + b] = 0;

	a = posix_memalign((void **)&tranMatrix_update, 128, (long long)feat_size * layer1_size * sizeof(float));
	if (tranMatrix_update == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix_update [a * feat_size + b] = 0;
//	CreateBinaryTree();
	BuildPara();
}

void FixedSy_ResetPara()
{
	int a, b;
	for(a = 0; a < layer1_size; a++)
		for (b = 0; b < feat_size; b ++)
			tranMatrix_delta [a * feat_size + b] = 0;
}
void FixedSy_ReadParaSy()
{
	char word[MAX_STRING];
	FILE *fin;
	int rows = 0, cols = 0;
	if (read_sy == 1){
		//read syn0;
		fin = fopen(syn0_best_file, "r");
		if (fin == NULL) {
			printf("ERROR: Syn0 file not found!\n");
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

			if (rows == 0 && cols == 0)
			{
				printf("Initial Epoch Loss: %s\n", word);
				iter_loss = std::stof(word);
			}
			if (rows > 0){
				if (cols > 0){
					syn0[(rows - 1) * layer1_size + cols - 1] = std::stof(word);
					//					if ((cols - 1)%300 == 0)
					//						printf("rows * layer1_size + cols :%lld \t, syn0 value:%f \n", (rows-1) * layer1_size + cols -1 , syn0[(rows-1) * layer1_size + cols -1]);
				}
			}
			cols ++;
		}
		fclose(fin);
		rows = 0; cols = 0;

		//read syn1;
		fin = fopen(syn1_best_file, "rb");
		if (fin == NULL) {
			printf("ERROR: Syn1 file not found!\n");
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
			if(rows > 0){
				if (cols > 0){
					syn1[(rows - 1) * layer1_size + cols - 1] = std::stof(word);
					//					if ((cols - 1)%300 == 0)
					//						printf("rows * layer1_size + cols :%lld \t, syn1 value:%f \n", (rows-1) * layer1_size + cols -1 , syn1[(rows-1) * layer1_size + cols -1]);
				}
				cols ++;
			}
		}
		fclose(fin);
		rows = 0; cols = 0;
	}
}

void FixedSy_TrainModel() {  //here different threads share variables. so they can implement the parallel processing.
	//Train the models first, get the word vectors, then save the word vectors or cluster the word vectors.
	long a, b;
	FILE *fo_best, *f1_best, *fo_loss;
	float epoch_loss = 0, aa_epoch_loss = 0, ab_epoch_loss = 0, ba_epoch_loss = 0, bb_epoch_loss = 0, last_epoch_loss = 0;
	double epoch_loss_best = 1000;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	time_t now_time = time(NULL);
	struct tm *local;
	printf("Fixed Syn Learning Rate: %f\t Scale %d\t Pair_size %lld \t Total size %lld \t \n", fixedsy_starting_alpha, scale, pair_size, total_pairs);

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
		pthread_create(&pt[a], NULL, FixedSy_compute_epoch_loss, (void *) &loss_struct[a]);
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
	fo_loss = fopen(fixed_sy_loss_file, "ab");
	fprintf(fo_loss, "Epoch No: %d, Loss: %lf\n", -1, epoch_loss);
	fclose(fo_loss);
	epoch_loss = 0; aa_epoch_loss = 0; ab_epoch_loss = 0; ba_epoch_loss = 0; bb_epoch_loss = 0;

	for (int e = 0; e < num_epoches; e ++)
	{
		fixedsy_alpha = fixedsy_starting_alpha * (1 - e / (float)(num_epoches + 1));
		if (fixedsy_alpha < fixedsy_starting_alpha * 0.0001) fixedsy_alpha = fixedsy_starting_alpha * 0.0001;
		cout << "===============================================" << endl;
		cout << "epoch no:  " <<e<<endl;
		//Record time.
		now_time = time(NULL);
		local = localtime(&now_time);
		cout << asctime(local);

		UpdateEpoch();

		//Record Time
		now_time = time(NULL);
		local = localtime(&now_time);
		cout << asctime(local);
		printf("Current Time: %s", asctime(local));

		//Compute loss before update.
		for(a = 0; a < num_threads; a++){
			loss_struct[a].loss_thread = &loss_thread[a];
			loss_struct[a].aa_loss_thread = &aa_loss_thread[a];
			loss_struct[a].ab_loss_thread = &ab_loss_thread[a];
			loss_struct[a].ba_loss_thread = &ba_loss_thread[a];
			loss_struct[a].bb_loss_thread = &bb_loss_thread[a];
			loss_struct[a].thread_id = a;
			pthread_create(&pt[a], NULL, FixedSy_compute_epoch_loss, (void *) &loss_struct[a]);
		}
		for (a = 0; a < num_threads; a ++) pthread_join(pt[a], NULL);

		for (int i = 0; i < num_threads; i++){
			epoch_loss += loss_thread[i];
			aa_epoch_loss += aa_loss_thread[i];
			ab_epoch_loss += ab_loss_thread[i];
			ba_epoch_loss += ba_loss_thread[i];
			bb_epoch_loss += bb_loss_thread[i];
		}

		epoch_loss = epoch_loss/total_pairs;
		aa_epoch_loss = aa_epoch_loss/total_pairs;
		ab_epoch_loss = ab_epoch_loss/total_pairs;
		ba_epoch_loss = ba_epoch_loss/total_pairs;
		bb_epoch_loss = bb_epoch_loss/total_pairs;

		printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", epoch_loss,
				aa_epoch_loss, ab_epoch_loss, ba_epoch_loss, bb_epoch_loss);
		fo_loss = fopen(fixed_sy_loss_file, "ab");
		fprintf(fo_loss, "Epoch No: %d, Loss: %lf\n", e, epoch_loss);
		fclose(fo_loss);

		if(epoch_loss_best > epoch_loss)
		{
			epoch_loss_best = epoch_loss;
			for (a = 0; a < layer1_size; a++) {
				for (b = 0; b < feat_size; b++){
					tranMatrix_best[a * feat_size + b] = tranMatrix[a * feat_size + b];
				}
			}
			fixsy_alpha_best = fixedsy_alpha;
		}
		if (abs(last_epoch_loss - epoch_loss) / epoch_loss < 0.01)
		{
			//save sy variable
			fo_best = fopen(iter_syn0_best_file, "wb");
			f1_best = fopen(iter_syn1_best_file, "wb");
			fprintf(fo_best, "%f %lld %d %f %f\n", iter_loss, vocab_size, layer1_size, fixedw_alpha_best, fixsy_alpha_best);
			fprintf(f1_best, "%f %lld %d %f %f\n", iter_loss, vocab_size, layer1_size, fixedw_alpha_best, fixsy_alpha_best);
			for (a = 0; a < vocab_size; a++) {
				fprintf(fo_best, "%s ", vocab[a].word);
				fprintf(f1_best, "%s ", vocab[a].word);
				for (b = 0; b < layer1_size; b++) 	{
					fprintf(fo_best, "%lf ", syn0_best[a * layer1_size + b]);
					fprintf(f1_best, "%lf ", syn1_best[a * layer1_size + b]);
				}
				fprintf(fo_best, "\n");
				fprintf(f1_best, "\n");
			}
			fclose(fo_best); fclose(f1_best);

			//save w variable
			fo_best = fopen(iter_w_best_file, "wb");
			fprintf(fo_best, "%f %lld %d %f %f\n", iter_loss, vocab_size, layer1_size, fixedw_alpha_best, fixsy_alpha_best);
			for (a = 0; a < layer1_size; a++) {
				for (b = 0; b < feat_size; b++){
					fprintf(fo_best, "%lf ", tranMatrix_best[a * feat_size + b]);
				}
				fprintf(fo_best, "\n");
			}
			fclose(fo_best);

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
