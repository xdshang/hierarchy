#ifndef _HDF5UTILS_H
#define _HDF5UTILS_H

#include <hdf5.h>

#define SINGLE
#ifdef SINGLE
typedef float Dtype;
#else
typedef double Dtype;
#endif

hid_t h5open(const char* file_name, const char* dataset_name,
    hsize_t ndim, const hsize_t* dims);

herr_t h5read(hid_t dataset, hsize_t ndim,
    const hsize_t* offset, const hsize_t* count, Dtype* buffer);

herr_t h5write(hid_t dataset, hsize_t ndim,
    const hsize_t* offset, const hsize_t* count, const Dtype* buffer);

herr_t h5close(hid_t dataset);

#endif