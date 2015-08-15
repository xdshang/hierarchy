#include "hdf5utils.h"

#ifdef SINGLE
#define H5TYPE H5T_NATIVE_FLOAT
#else
#define H5TYPE H5T_NATIVE_DOUBLE
#endif

hid_t h5open(const char* file_name, const char* dataset_name,
    hsize_t ndim, const hsize_t* dims) {
  H5E_auto2_t old_func;
  void *old_client_data;
  hid_t file, dataset, filespace;
  // mute the error auto-printing
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  file = H5Fcreate(file_name, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
  // if file exists, just open it
  if (file < 0) {
    file = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
      return file;
    }
  }

  dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
  if (dataset < 0) {
    filespace = H5Screate_simple(ndim, dims, NULL);
    dataset = H5Dcreate2(file, dataset_name, H5TYPE, filespace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // TODO: initialize the dataset
  }
  else {
    hsize_t current_dims[ndim], i;
    // check type consistence
    if (H5Tequal(H5Dget_type(dataset), H5TYPE) <= 0) {
      return -1;
    }
    // check dimension consistence
    filespace = H5Dget_space(dataset);
    if (H5Sget_simple_extent_dims(filespace, current_dims, NULL) != ndim) {
      return -1;
    }
    for (i = 0; i < ndim; ++i) {
      if (current_dims[i] != dims[i]) {
        return -1;
      }
    }
  }
  // restore the error handler
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
  H5Sclose(filespace);
  return dataset;
}

herr_t _h5rw_helper(hid_t dataset, hsize_t ndim,
    const hsize_t* offset, const hsize_t* count,
    hid_t* filespace, hid_t* memspace) {
  herr_t status;

  *filespace = H5Dget_space(dataset);
  if (H5Sget_simple_extent_ndims(*filespace) != ndim) {
    return -1;
  }
  *memspace = H5Screate_simple(ndim, count, NULL);

  if ((status = H5Sselect_hyperslab(*filespace, H5S_SELECT_SET, 
      offset, NULL, count, NULL)) < 0) {
    return status;
  }

  return 0;
}

herr_t h5read(hid_t dataset, hsize_t ndim,
    const hsize_t* offset, const hsize_t* count, Dtype* buffer) {
  hid_t filespace, memspace;
  herr_t status;

  if ((status = _h5rw_helper(dataset, ndim, offset, count,
      &filespace, &memspace)) < 0) {
    return status;
  }
  if ((status = H5Dread(dataset, H5TYPE, memspace, filespace, 
      H5P_DEFAULT, buffer)) < 0) {
    return status;
  }

  H5Sclose(filespace);
  H5Sclose(memspace);
  return 0;
}

herr_t h5write(hid_t dataset, hsize_t ndim,
    const hsize_t* offset, const hsize_t* count, const Dtype* buffer) {
  hid_t filespace, memspace;
  herr_t status;

  if ((status = _h5rw_helper(dataset, ndim, offset, count,
      &filespace, &memspace)) < 0) {
    return status;
  }
  if ((status = H5Dwrite(dataset, H5TYPE, memspace, filespace, 
      H5P_DEFAULT, buffer)) < 0) {
    return status;
  }

  H5Sclose(filespace);
  H5Sclose(memspace);
  return 0;
}

herr_t h5close(hid_t dataset) {
  hid_t file;
  herr_t status;

  if ((file = H5Iget_file_id(dataset)) < 0) {
    return file;
  }
  if ((status = H5Dclose(dataset)) < 0) {
    return status;
  }
  if ((status = H5Fclose(file)) < 0) {
    return status;
  }

  return 0;
}