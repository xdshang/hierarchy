#include "sync_param.h"

#include <stdlib.h>

#define MAX_NUM_PARAM 5
#define DATASET_NAME "param"

typedef struct _Handle {
  hid_t h5file;
  int num, dim;
  // 0 - in disk; 1 - in memory and clean; 2 - dirty
  // 3 - clean and to be removed; 4 - dirty and to be removed
  char *status;
  int *rm_list, rm_size; // index to be removed
  int *curr_list, curr_size; // current index in use
  int *hash, hash_size; // hash to locate mem
  Dtype *mem;
} Handle;

Handle _handle[MAX_NUM_PARAM];
int _hid = 0;

int create_sync_param(const char *fname, const int num, const int dim) {
  int hid = _hid;
  ++_hid;
  // open a new hdf5 file
  // check the dimension
  return hid;
}

int destroy_sync_param(const int hid);

const Dtype* sync_param(const int hid, const int id);

Dtype* mutable_sync_param(const int hid, const int id);

int compare(const void *a, const void *b) {
  return (int)*a - (int)*b;
}

int prefetch_sync_param(const int hid, int size, int* list) {
  int i, j, p, fet_size;
  // sort list in ascending order
  qsort(list, size, sizeof(int), compare);
  // remove duplicated data
  for (i = 0, j = 0, p = -1; i < size; ++i) {
    if (list[i] != p) {
      list[j] = list[i];
      p = list[i];
      ++j;
    }
  }
  size = j;
  // if the datum is already in memory, move it to the back part
  for (i = 0, j = size - 1; i < j;) {
    for (; _handle[hid].status[list[i]]; ++i) {}
    for (; !_handle[hid].status[list[j]]; --j) {}
    if (i < j) {
      p = list[i];
      list[i] = list[j];
      list[j] = p;
    }
  }
  fet_size = j + 1;
  // if the datum is to be removed, cancel it and add it to current list
  for (i = fet_size; i < size; ++i) {
    if (_handle[hid].status[list[i]] < 0) {
      _handle[hid].status[list[i]] = -_handle[hid].status[list[i]];
      // add to current list
      _handle[hid].curr_list[_handle[hid].curr_size++] = list[i];
    }
  }
  // perform removing
  for (i = 0; i < _handle[hid].rm_size; ++i) {
    p = _handle[hid].rm_list[i];
    if (_handle[hid].status[p] < 0) {
      if (_handle[hid].status[p] == -2) {
        // TODO: write back to hdf5 file
      }
      // TODO: remove from hashtable
    }
  }
  // perform prefetching
  for (i = 0; i < fet_size; ++i) {
    _handle[hid].status[list[i]] = 1;
    // TODO: get hash
    // TODO: read from hdf5 file
  }
  // generate rm_list for next iteration
  for (i = 0; i < ) {

  }
  return 0;
}