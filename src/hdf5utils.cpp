#include "caffe/util/hdf5utils.hpp"

#include <cassert>

namespace caffe{

template <typename Dtype>
void h5read(const char* file_name, const char* dataset_name,
        hsize_t ndim, const hsize_t* offset, const hsize_t* count, 
        Dtype* buffer){
    herr_t status;
    hid_t file = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    hid_t datatype  = H5Dget_type(dataset);
    size_t size = H5Tget_precision(datatype);
    assert(size == sizeof(Dtype) * 8);

    hid_t dataspace = H5Dget_space(dataset);
    hid_t memspace = H5Screate_simple(ndim, count, NULL);
    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
    status = H5Dread(dataset, datatype, memspace, dataspace, H5P_DEFAULT, buffer);

    status = H5Sclose(dataspace);
    status = H5Tclose(datatype);
    status = H5Dclose(dataset);
    status = H5Fclose(file);
}

template void h5read<float>(const char*, const char*, hsize_t, 
		const hsize_t*, const hsize_t*, float*);
template void h5read<double>(const char*, const char*, hsize_t, 
		const hsize_t*, const hsize_t*, double*);

template <typename Dtype>
void h5write(const char* file_name, const char* dataset_name,
        hsize_t ndim, const hsize_t* offset, const hsize_t* count, 
        const Dtype* buffer){
	herr_t status;
	hid_t file, dataset, datatype, dataspace;

	if((file = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT)) < 0){
		file = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if(sizeof(Dtype) == sizeof(float))
			datatype = H5T_NATIVE_FLOAT;
		else if(sizeof(Dtype) == sizeof(double))
			datatype = H5T_NATIVE_DOUBLE;
		else
			assert(0);

		hsize_t init_dims[ndim];
		hsize_t init_maxdims[ndim];
		hsize_t chunk_dims[ndim];
		init_dims[0] = 0;
		init_maxdims[0] = H5S_UNLIMITED;	// unlimited along the last dimension
		chunk_dims[0] = 1000;	// set it accordingly
		for(int i = 1; i < ndim; ++i){
			init_dims[i] = count[i];
			init_maxdims[i] = init_dims[i];
			chunk_dims[i] = init_maxdims[i];
		}
		dataspace = H5Screate_simple(ndim, init_dims, init_maxdims);

		hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
		status = H5Pset_chunk(prop, ndim, chunk_dims);

		dataset = H5Dcreate2(file, dataset_name, datatype, dataspace,
                H5P_DEFAULT, prop, H5P_DEFAULT);

		status = H5Pclose(prop);
	}
	else{
		dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
		datatype = H5Dget_type(dataset);
	    size_t size = H5Tget_precision(datatype);
	    assert(size == sizeof(Dtype) * 8);
	    dataspace = H5Dget_space(dataset);
	}

	hsize_t current_dims[ndim];
	assert(ndim == H5Sget_simple_extent_dims(dataspace, current_dims, NULL));
	if(current_dims[0] < offset[0] + count[0]){
		current_dims[0] = offset[0] + count[0];
		status = H5Dset_extent(dataset, current_dims);
		dataspace = H5Dget_space(dataset);
	}

    hid_t memspace = H5Screate_simple(ndim, count, NULL);
    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
    status = H5Dwrite(dataset, datatype, memspace, dataspace, H5P_DEFAULT, buffer);

    status = H5Sclose(dataspace);
    status = H5Tclose(datatype);
    status = H5Dclose(dataset);
    status = H5Fclose(file);
}

template void h5write<float>(const char*, const char*, hsize_t, 
		const hsize_t*, const hsize_t*, const float*);
template void h5write<double>(const char*, const char*, hsize_t, 
		const hsize_t*, const hsize_t*, const double*);

}