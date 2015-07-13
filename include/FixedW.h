/*
 * FixedW.h
 *
 *  Created on: Oct 28, 2014
 *      Author: lms-gpu
 */
#include "ValueDef.h"
extern int read_w, read_sy_delta;
extern float fixedW_alpha, fixedW_starting_alpha;
extern char fixed_w_loss_file[MAX_STRING];

extern void *FixedW_compute_epoch_loss(void *id);
void *FixedW_TrainModelThread(void *id) ;
void FixedW_TrainModel();
void FixedW_InitNet();
void FixedW_ResetPara();
