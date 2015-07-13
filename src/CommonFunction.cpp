/*
 * CommonFunction.cpp
 *
 *  Created on: Oct 29, 2014
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
#include <dirent.h>
using namespace std;

#include "CommonFunction.h"
#include "ValueDef.h"
#include "Word2Vec.h"
#include "FixedW.h"
#include "FixedSy.h"

void ReadSy_Delta(){
	char word[MAX_STRING];
	FILE *fin;
	int rows = 0, cols = 0;
	cout << "read_sy_delta" << read_sy_delta <<endl;
	if (read_sy_delta == 1){
		//read syn0;
		fin = fopen(syn0_delta_file, "r");
		if (fin == NULL) {
			printf("ERROR: Syn0 delta file not found!\n");
			exit(1);
		}
		while (1) {
			ReadWord(word, fin);
			if (feof(fin)) break;  // file end, break out.
			if(strcmp(word, (char *)"</s>") == 0){
				rows ++;
				cols = 0;
				continue;
			}
			if (rows > 0){
				if (cols > 0){
					syn0_delta[(rows - 1) * layer1_size + cols - 1] = std::stof(word);
				}
				cols ++;
			}
		}
		fclose(fin);
		rows = 0; cols = 0;

		//read syn1;
		fin = fopen(syn1_delta_file, "rb");
		if (fin == NULL) {
			printf("ERROR: Syn1 delta file not found!\n");
			exit(1);
		}
		while (1) {
			ReadWord(word, fin);
			if (feof(fin)) break;  // file end, break out.
			if(strcmp(word, (char *)"</s>") == 0){
				rows ++;
				cols = 0;
				continue;
			}
			if(rows > 0){
				if (cols > 0){
					syn1_delta[(rows - 1) * layer1_size + cols - 1] = std::stof(word);
					//					if ((cols - 1)%299 == 0)
					//						printf("rows * layer1_size + cols :%d \t, syn1 value:%f \n", (rows-1) * layer1_size + cols -1 , syn1_delta[(rows-1) * layer1_size + cols -1]);
				}
				cols ++;
			}
		}
		fclose(fin);
		rows = 0; cols = 0;
	}
}

/*
 * Common Iteration first.
 */
void ResetPara_sy(){
	int a, b = 0;
	//reset sy
	for (a = 0; a < vocab_size; a++) {
		for (b= 0; b < layer1_size; b++){
			syn0[a * layer1_size + b] = syn0_best[a * layer1_size + b];
			syn1[a * layer1_size + b] = syn1_best[a * layer1_size + b];
			//			syn0_delta[a * layer1_size + b] = syn0_delta_best[a * layer1_size + b];
			//			syn1_delta[a * layer1_size + b] = syn1_delta_best[a * layer1_size + b];
		}
	}
}

void InitSyn_best(){
	int a, b = 0;
	//init sy_best
	for (a = 0; a < vocab_size; a++) {
		for (b= 0; b < layer1_size; b++){
			syn0_best[a * layer1_size + b] = syn0[a * layer1_size + b];
			syn1_best[a * layer1_size + b] = syn1[a * layer1_size + b];
			//			syn0_delta_best[a * layer1_size + b] = syn0_delta[a * layer1_size + b];
			//			syn1_delta_best[a * layer1_size + b] = syn1_delta[a * layer1_size + b];
		}
	}
}
/*
 *reset delta
 */
void ResetDelta(){
	int a, b = 0;
	for (a = 0; a < vocab_size; a++) {
		for (b= 0; b < layer1_size; b++){
			syn0_delta[a * layer1_size + b] = 0;
			syn1_delta[a * layer1_size + b] = 0;
		}
	}
	for (a = 0; a < layer1_size; a++) {
		for (b = 0; b < feat_size; b++){
			tranMatrix_delta[a * feat_size + b] = 0;
		}
	}
}


void ResetPara_w(){
	//reset w.
	int a, b = 0;
	fixedW_alpha = fixedw_alpha_best;
	for (a = 0; a < layer1_size; a++) {
		for (b = 0; b < feat_size; b++){
			tranMatrix[a * feat_size + b] = tranMatrix_best[a * feat_size + b];
			//			tranMatrix_delta[a * feat_size + b] = tranMatrix_delta_best[a * feat_size + b];
		}
	}
}


/*
 * Read Image Index from Files.
 */
void readImageIndex (char imageIndexPath[MAX_STRING]){

	ifstream infile(imageIndexPath);

	string line;
	string a;
	int b;
	int count = 0;
	while (infile >> a >> b){
		//		cout << a << " "<< b << endl;
		count ++;
		if(count > 100000){
			printf("ImageName:  %s, ImageIndex: %d. ", a.c_str(), b);
			count = 0;
		}
		image_index[a] = b;
	}
}

void readFeat_folder(char *featfolder){
	string line;
	int a, b;
	float c;

	int offset = 0;
	int seen_a = -1;
	int temp = 0;
	int count = 0;

	struct dirent *ptr;
	DIR *dir;
	dir=opendir(featfolder);
	vector<string> files;
	while((ptr=readdir(dir))!=NULL)
	{

		if(ptr->d_name[0] == '.')
			continue;
		cout << ptr->d_name << endl;
		files.push_back(ptr->d_name);
	}
	unsigned int i;
	for (i = 0; i < files.size(); ++i)
	{
	//	cout << (string(featfolder) + files[i]) << endl;
		ifstream infile((string(featfolder) + files[i]).c_str());
		while (infile >> a >> b >> c){
			count ++;
			if(count > 20000000){
				printf("ImageIndex:  %d, Image NonzeroIndex: %d, Image NonzeroValue: %f ", a, b, c);
				count =0;
			}

			if (c == 0){
//				cout << "error feat:  " << a << "   " << b <<endl;
			}
			temp = nonzeroIdx[a - 1];
			if(seen_a != a){
				offset = 0;
				seen_a = a;
			}
			else
				offset ++;

			Idx[temp + offset] = b;
			Val[temp + offset] = c;
		}
	}
}

/*
 * Idx: get the image feature index
 * Val: get the image feature value
 */
void readFeat(char featFilePath[MAX_STRING])
{
	string line;
	int a, b;
	float c;

	int offset = 0;
	int seen_a = -1;
	int temp = 0;
	int count = 0;
	ifstream infile(featFilePath);
	while (infile >> a >> b >> c){
		count ++;
		if(count > 20000000){
			//			cout << "ImageIndex: "<< a <<  "Nonzero Index:" << b << "Nonzero Value: " << c << endl; //Check Whether Loading Right.
			printf("ImageIndex:  %d, Image NonzeroIndex: %d, Image NonzeroValue: %f ", a, b, c);
			count =0;
		}

		if (c == 0){
			cout << "error feat" <<endl;
		}
		temp = nonzeroIdx[a - 1];
		if(seen_a != a){
			offset = 0;
			seen_a = a;
		}
		else
			offset ++;

		Idx[temp + offset] = b;
		Val[temp + offset] = c;
	}
}
/*
 * countNonzero in a folder
 */
void countNonzero_folder(char featfolder[MAX_STRING]){
	string line;
	int a, b, seen_a, nonzero;
	float c;

	seen_a = -1;
	nonzero = 0;
	nonzeroIdx[0] = 0;


	struct dirent *ptr;
	DIR *dir;
	dir=opendir(featfolder);
	vector<string> files;
	while((ptr=readdir(dir))!=NULL)
	{

		if(ptr->d_name[0] == '.')
			continue;
		cout << ptr->d_name << endl;
		files.push_back(ptr->d_name);
	}

	unsigned int i;
	for (i = 0; i < files.size(); ++i)
	{
		//cout << (string(featfolder) + files[i]) << endl;
		ifstream infile((string(featfolder) + (files[i])).c_str());
		while(infile >> a >> b >> c)
		{
			if(seen_a != a)
			{
				if(seen_a == -1)
				{
					seen_a = a;
					nonzero ++;
					continue;
				}
				nonzeroLen[seen_a - 1] = nonzero; //The last Image
				nonzeroIdx[a - 1] = nonzeroIdx[seen_a - 1] + nonzero; //Current Image.
				seen_a = a;
				nonzero = 1;
			}
			else{
				nonzero ++;
			}
		}
	}
	nonzeroLen[a - 1] = nonzero;
	closedir(dir);
}
/*
 *  nonzeroIdx: record first position nonzero of an Image. from 0.
 *  nonzeroLen: record feature nonzero length of Image.
 */
void countNonzero (char featFilePath[MAX_STRING])
{
	string line;
	int a, b, seen_a, nonzero;
	float c;

	seen_a = -1;
	nonzero = 0;
	nonzeroIdx[0] = 0;
	ifstream infile(featFilePath);
	while(infile >> a >> b >> c)
	{
		if(seen_a != a)
		{
			if(seen_a == -1)
			{
				seen_a = a;
				nonzero ++;
				continue;
			}
			nonzeroLen[seen_a - 1] = nonzero; //The last Image
			nonzeroIdx[a - 1] = nonzeroIdx[seen_a - 1] + nonzero; //Current Image.
			seen_a = a;
			nonzero = 1;
		}
		else
		{
			nonzero ++;
		}
	}
	nonzeroLen[a - 1] = nonzero;
}
/** detecting whether base is ends with str
 */
bool endsWith (char* base, const char* str) {
	int blen = strlen(base);
	int slen = strlen(str);
	return (blen >= slen) && (0 == strcmp(base + blen - slen, str));
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

	int wordstamp = 0;
	string ends = "jpg";
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
	cout<< "pair_size:\t" << pair_size<<endl;
	fclose(fin);
}
