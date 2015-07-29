#ifndef _HDF5UTILS_HPP_
#define _HDF5UTILS_HPP_

#include <hdf5.h>

namespace caffe{

template <typename Dtype>
void h5read(const char* file_name, const char* dataset_name,
        hsize_t ndim, const hsize_t* offset, const hsize_t* count, 
        Dtype* buffer);

template <typename Dtype>
void h5write(const char* file_name, const char* dataset_name,
        hsize_t ndim, const hsize_t* offset, const hsize_t* count, 
        const Dtype* buffer);

}

#endif