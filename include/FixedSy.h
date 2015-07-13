/*
 * FixedSy.h
 *
 *  Created on: Oct 28, 2014
 *      Author: lms-gpu
 */
// #include "ImSoftMaxVector.h"
#include "ValueDef.h"

extern int read_sy;
extern char fixed_sy_loss_file[MAX_STRING];
extern float fixedsy_alpha, fixedsy_starting_alpha;

//Cpp dependent
void *GetEachPairUpdate(void *threadarg);
void UpdateEpoch();
void *compute_epoch_loss (void *id);
void InitNet();
void FixedSy_PrintLoss();
void FixedSy_ReadParaSy();
void FixedSy_TrainModel();
void FixedSy_ResetPara();
void *check_load(void *id);

