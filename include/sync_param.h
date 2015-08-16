#ifndef _SYNC_PARAM_H
#define _SYNC_PARAM_H

// definition of Dtype
#include "hdf5utils.h"

// the number of bits to represent slot id
#define SLOT_BITS 10
#define MAX_NUM_SLOT (1 << SLOT_BITS)
// slots contain the current list and the prefetching list
#define MAX_NUM_LIST (MAX_NUM_SLOT / 2)
#define MAX_NUM_HANDLE 5

typedef struct _PrefetchArgs {
  int hid, size, status;
  int* list;
} PrefetchArgs;

int create_sync_param(const char *fname, const int num, const int dim);

int destroy_sync_param(const int hid);

const Dtype* sync_param(const int hid, const int id);

Dtype* mutable_sync_param(const int hid, const int id);

// pthread prototype
void *prefetch_sync_param(void *args);

// Debug Interface

unsigned int* get_status(const int hid);

int get_rm_size(const int hid);

int* get_rm_list(const int hid);

int get_curr_size(const int hid);

int* get_curr_list(const int hid);

int get_slot_ptr(const int hid);

int* get_slot(const int hid);

#endif
