/*
 * Main.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: lms-gpu
 */

//Evaluation
//Main to call the two functions.
//Transfer from one function to another, the variable value changes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
// #include "ImSoftMaxVector.h"
#include "FixedSy.h"
#include "FixedW.h"
#include "Word2Vec.h"
// #include "Main.h"
#include "ValueDef.h"
#include "CommonFunction.h"

using namespace std;

void Run()
{
	//Init operations.
	time_t now_time = time(NULL);
	struct tm *local;
	local = localtime(&now_time);
	cout << asctime(local);
	if (read_vocab_file[0] != 0) 
    ReadVocab(); 
  else 
    LearnVocabFromTrainFile();
	if ((save_vocab_file)[0] != 0) 
    SaveVocab();
  LearnPairsFromFile();
//	BuildPara();
//	exit(0);
	cout << "-----------------------------------------------------"<<endl;
	readImageIndex(image_index_file);
	cout << feat_folder << endl;//read image index
	countNonzero_folder(feat_folder); //count feature
	cout << "-----------------------------------------------------"<<endl;
	readFeat_folder(feat_folder); //read feature
	cout << "-----------------------------------------------------"<<endl;

	InitNet(); //InitNet.
	now_time = time(NULL);
	local = localtime(&now_time);
	cout << asctime(local);

	read_sy = 1;
	FixedSy_ReadParaSy(); //Tested read syn0, syn1.

	fixedsy_starting_alpha = 0.05;
	fixedW_starting_alpha = 0.001;

	//Init syn0_best = syn0.
	InitSyn_best();
	FixedSy_TrainModel(); //update W.
//	system("cp ../468/*.txt ../468/1_iter");
//	while(true){
//		//reset delta.
//		ResetDelta();
//		iteration++;
//		cout << "--------------------------------------------------update syn0_tag and inner node start! --------------------------------------------" <<endl;
//		ResetPara_w(); //get best w para. update w_delta
//		FixedW_TrainModel();
//		cout << "--------------------------------------------------update syn0_tag and inner node end! --------------------------------------------" <<endl;
//
//		cout << "--------------------------------------------------update W start--------------------------------------------" <<endl;
//		ResetPara_sy(); //get best syn & syn delta. update syn_delta
//		FixedSy_TrainModel(); //update W.
//		cout << "--------------------------------------------------update W end--------------------------------------------" <<endl;
//		string s1 = "mkdir ../group1/";
//		std::string s = to_string(iteration);
//		string s2 = s1+s;
//		system(s2.c_str());
//
//		string s3 = "../group1/iteration/";
//		string s3 =
//		system("cp ../group1/*.txt " + s2);
//		system("cp ../468/*.txt ../468/last_iter");
//		cout << "---------------------------Iteration " << iteration << "----------------loss  " <<iter_loss_best << "------------"<<endl;
//		if (iteration > MaxIteration){
//			break;
//		}
//	}
}

int* nonzeroIdx; //nonzeroIdx[i] is the starting idx of array Idx
int* nonzeroLen; //nonzeroLen[i] is the #non zero entries of image i.
float* Val;
int* Idx;

int main(int argc, char **argv) {
	int i;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	syn0_best_file[0] = 0;
	char iter_loss_file[MAX_STRING];
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-feat", argc, argv)) > 0) strcpy(feat_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-feat-folder", argc, argv)) > 0) strcpy(feat_folder, argv[i + 1]);
	if ((i = ArgPos((char *)"-image-index", argc, argv)) > 0) strcpy(image_index_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) num_epoches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-syn0-best", argc, argv)) > 0) strcpy(syn0_best_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-syn1-best", argc, argv)) > 0) strcpy(syn1_best_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-syn0-temp", argc, argv)) > 0) strcpy(syn0_temp_file, argv[i + 1]);

	if ((i = ArgPos((char *)"-iter-syn0", argc, argv)) > 0) strcpy(iter_syn0_best_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-iter-syn1", argc, argv)) > 0) strcpy(iter_syn1_best_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-iter-w", argc, argv)) > 0) strcpy(iter_w_best_file, argv[i + 1]);

	if ((i = ArgPos((char *)"-fixed-w-loss", argc, argv)) > 0) strcpy(fixed_w_loss_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-fixed-sy-loss", argc, argv)) > 0) strcpy(fixed_sy_loss_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-iter-loss", argc, argv)) > 0) strcpy(iter_loss_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-tree-file", argc, argv)) > 0) strcpy(tree_file, argv[i + 1]);

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
	for (i = 0; i < EXP_TABLE_SIZE + 1; i++) {
		expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	nonzeroIdx = new int[(size_t)NUM_IMG]; //nonzeroIdx[i] is the starting idx of array Idx
	nonzeroLen = new int[(size_t)NUM_IMG]; //nonzeroLen[i] is the #non zero entries of image i.
	Val = new float[(size_t)NUM_NONZERO];
	Idx = new int[(size_t)NUM_NONZERO];
	Run();
	return 0;
}
