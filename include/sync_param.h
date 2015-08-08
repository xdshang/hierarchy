#ifndef _SYNC_PARAM_H
#define _SYNC_PARAM_H

typedef float Dtype;

int create_sync_param(const char *fname, const int num, const int dim);

int destroy_sync_param(const int hid);

const Dtype* sync_param(const int hid, const int id);

Dtype* mutable_sync_param(const int hid, const int id);

int prefetch_sync_param(const int hid, int size, int* list);

#endif
