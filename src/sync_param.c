#include "sync_param.h"

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <pthread.h>

#define DATASET_NAME "param"
#define NDIM 2
#define SLOT_MASK (MAX_NUM_SLOT - 1)
// define errors
#define NOT_ENOUGH_HANDLE_ERROR -1
#define INVALID_HID_ERROR -2
#define INVALID_ID_ERROR -3
#define EXCEED_MAX_NUM_LIST_ERROR -4
#define EXCEED_MAX_NUM_SLOT_ERROR -5
#define HDF5_OPEN_ERROR -6
#define HDF5_RW_ERROR -7
#define HDF5_CLOSE_ERROR -8

typedef struct handle {
  hid_t h5id;
  int num, dim;
  // slot_id | [to_fetch] [to_rm] [is_dirty] [in_mem]
  unsigned int *status; 
  // index to be removed
  int *rm_list, rm_size;
  // current index in use
  int *curr_list, curr_size;
  int *slot, slot_ptr;
  Dtype *mem;
} Handle;

Handle handle[MAX_NUM_HANDLE];
int handle_id[MAX_NUM_HANDLE];
int handle_ptr = MAX_NUM_HANDLE;

int _valid_hid(const int hid) {
  int i;
  // check if it is within the range
  if (hid < 0 || hid >= MAX_NUM_HANDLE) {
    return 0;
  }
  // check if it is not used
  for (i = 0; i <= handle_ptr; ++i) {
    if (handle_id[i] == hid) {
      return 0;
    }
  }

  return 1;
}

int create_sync_param(const char *fname, const int num, const int dim) {
  hsize_t dims[NDIM] = {num, dim};
  int i, hid; 
  // initialize handle pool
  if (handle_ptr >= MAX_NUM_HANDLE) {
    for (i = 0; i < MAX_NUM_HANDLE; ++i) {
      handle_id[i] = i;
    }
    handle_ptr = MAX_NUM_HANDLE - 1;
  }
  if (handle_ptr == -1) {
    return NOT_ENOUGH_HANDLE_ERROR;
  }
  hid = handle_id[handle_ptr--];
  // open hdf5 file and check dimensions
  handle[hid].h5id = h5open(fname, DATASET_NAME, NDIM, dims);
  if (handle[hid].h5id < 0) {
    return HDF5_OPEN_ERROR;
  }
  handle[hid].num = num;
  handle[hid].dim = dim;
  handle[hid].status = (unsigned int*)calloc(num, sizeof(unsigned int));
  handle[hid].rm_size = 0;
  handle[hid].rm_list = (int*)malloc(MAX_NUM_LIST * sizeof(int));
  handle[hid].curr_size = 0;
  handle[hid].curr_list = (int*)malloc(MAX_NUM_LIST * sizeof(int));
  handle[hid].slot_ptr = MAX_NUM_SLOT - 1;
  handle[hid].slot = (int*)malloc(MAX_NUM_SLOT * sizeof(int));
  handle[hid].mem = (Dtype*)malloc(MAX_NUM_SLOT * dim * sizeof(int));

  for (i = 0; i < MAX_NUM_SLOT; ++i) {
    handle[hid].slot[i] = i;
  }

  return hid;
}

int destroy_sync_param(const int hid) {
  hsize_t offset[NDIM] = {-1, 0};
  hsize_t count[NDIM] = {1, handle[hid].dim};
  int i, j, stat, err = 0;

  if (!_valid_hid(hid)) {
    return INVALID_HID_ERROR;
  }
  // write back the remained ones
  for (i = 0; i < handle[hid].curr_size; ++i) {
    handle[hid].rm_list[handle[hid].rm_size++] = handle[hid].curr_list[i];
  }
  for (i = 0; i < handle[hid].rm_size; ++i) {
    offset[0] = handle[hid].rm_list[i];
    stat = handle[hid].status[offset[0]];
    j = stat >> 4;
    if ((stat & 0x2) && (h5write(handle[hid].h5id, NDIM, offset, count, 
        handle[hid].mem + j * handle[hid].dim) < 0)) {
      err = HDF5_RW_ERROR;
    }
  }
  // close hdf5 file
  if (h5close(handle[hid].h5id) < 0) {
    err = HDF5_CLOSE_ERROR;
  }
  free(handle[hid].status);
  free(handle[hid].rm_list);
  free(handle[hid].curr_list);
  free(handle[hid].slot);

  memset(&(handle[hid]), 0, sizeof(Handle));
  handle_id[++handle_ptr] = hid;

  return err;
}

const Dtype* sync_param(const int hid, const int id) {
  if (_valid_hid(hid) && (handle[hid].status[id] & 0x1)) {
    int p = handle[hid].status[id] >> 4;
    return (const Dtype*)handle[hid].mem + p * handle[hid].dim;
  }
  return NULL;
}

Dtype* mutable_sync_param(const int hid, const int id) {
  if (_valid_hid(hid) && (handle[hid].status[id] & 0x1)) {
    int p = handle[hid].status[id] >> 4;
    handle[hid].status[id] |= 0x2;
    return handle[hid].mem + p * handle[hid].dim;
  }
  return NULL;
}

void *prefetch_sync_param(void *args) {
  int i, j, p, stat, fet_size = 0;
  PrefetchArgs* fet_args = (PrefetchArgs*)args;

  if (!_valid_hid(fet_args->hid)) {
    fet_args->status = INVALID_HID_ERROR;
    pthread_exit(NULL);
  }
  // reduce the fetching list
  for (i = 0; i < fet_args->size; ++i) {
    p = fet_args->list[i];
    if (p < 0 || p >= handle[fet_args->hid].num) {
      fet_args->status = INVALID_ID_ERROR;
      pthread_exit(NULL);
    }
    stat = handle[fet_args->hid].status[p];
    // if it is already marked to be feteched
    if (!(stat & 0x8)) {
      // if it is alreaddy in memory
      if (stat & 0x1) {
        // if it is to be removed
        if (stat & 0x4) {
          stat &= (SLOT_MASK << 4) | 0xb;
          handle[fet_args->hid].curr_list[handle[fet_args->hid].curr_size++] = p;
        }
      }
      else {
        fet_args->list[fet_size++] = p;
      }
      stat |= 0x8;
      handle[fet_args->hid].status[p] = stat;
    }
  }
  if (fet_size > MAX_NUM_LIST) {
    fet_args->status = EXCEED_MAX_NUM_LIST_ERROR;
    pthread_exit(NULL);
  }
  // perform removing
  for (i = 0; i < handle[fet_args->hid].rm_size; ++i) {
    p = handle[fet_args->hid].rm_list[i];
    stat = handle[fet_args->hid].status[p];
    // check if it is cancelled by the above procedure
    if (stat & 0x4) {
      j = stat >> 4;
      // write back to hdf5 file if it is dirty
      if (stat & 0x2) {
        hsize_t offset[NDIM] = {p, 0};
        hsize_t count[NDIM] = {1, handle[fet_args->hid].dim};
        if (h5write(handle[fet_args->hid].h5id, NDIM, offset, count, 
            handle[fet_args->hid].mem + j * handle[fet_args->hid].dim) < 0) {
          fet_args->status = HDF5_RW_ERROR;
          pthread_exit(NULL);
        }
      }
      // release the slot
      handle[fet_args->hid].slot[++handle[fet_args->hid].slot_ptr] = j;
      // clear status
      handle[fet_args->hid].status[p] = 0x0;
    }
  }
  // generate rm_list for the next iteration
  handle[fet_args->hid].rm_size = 0;
  for (i = 0, j = 0; i < handle[fet_args->hid].curr_size; ++i) {
    p = handle[fet_args->hid].curr_list[i];
    if (handle[fet_args->hid].status[p] & 0x8) {
      handle[fet_args->hid].status[p] &= (SLOT_MASK << 4) | 0x7;
      handle[fet_args->hid].curr_list[j] = p;
      ++j;
    }
    else {
      handle[fet_args->hid].rm_list[handle[fet_args->hid].rm_size++] = p;
      handle[fet_args->hid].status[p] |= 0x4;
    }
  }
  handle[fet_args->hid].curr_size = j;
  // append prefetch list and perform prefetching
  for (i = 0; i < fet_size; ++i) {
    // -1 means to be determined
    hsize_t offset[NDIM] = {-1, 0};
    hsize_t count[NDIM] = {1, handle[fet_args->hid].dim};
    // allocate a slot
    if (handle[fet_args->hid].slot_ptr == -1) {
      fet_args->status = EXCEED_MAX_NUM_SLOT_ERROR;
      pthread_exit(NULL);
    }
    p = fet_args->list[i];
    j = handle[fet_args->hid].slot[handle[fet_args->hid].slot_ptr--];
    handle[fet_args->hid].status[p] = ((unsigned int)j << 4) | 0x1;
    handle[fet_args->hid].curr_list[handle[fet_args->hid].curr_size++] = p;
    // read from hdf5 file
    offset[0] = p;
    if (h5read(handle[fet_args->hid].h5id, NDIM, offset, count, 
        handle[fet_args->hid].mem + j * handle[fet_args->hid].dim) < 0) {
      fet_args->status = HDF5_RW_ERROR;
      pthread_exit(NULL);
    }
  }

  fet_args->status = 0;
  pthread_exit(NULL);
}

// Debug Interface

unsigned int* get_status(const int hid) {
  return handle[hid].status;
}

int get_rm_size(const int hid) {
  return handle[hid].rm_size;
}

int* get_rm_list(const int hid) {
  return handle[hid].rm_list;
}

int get_curr_size(const int hid) {
  return handle[hid].curr_size;
}

int* get_curr_list(const int hid) {
  return handle[hid].curr_list;
}

int get_slot_ptr(const int hid) {
  return handle[hid].slot_ptr;
}

int* get_slot(const int hid) {
  return handle[hid].slot;
}