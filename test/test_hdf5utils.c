#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <check.h>

#include "hdf5utils.h"

#ifdef SINGLE
#define H5TYPE H5T_NATIVE_FLOAT
#else
#define H5TYPE H5T_NATIVE_DOUBLE
#endif

#define H5F_NAME "h5test_data.h5"
#define DATASET_NAME "param"

const hsize_t ndim = 2, chn = 3;
const hsize_t dims[] = {5, 3};
const hsize_t chl[] = {3, 0, 4};

START_TEST(test_h5open_close) {
  hid_t dataset;
  Dtype mem[dims[0] * dims[1]];
  int i, j;

  dataset = h5open(H5F_NAME, DATASET_NAME, ndim, dims);
  ck_assert_msg(dataset >= 0, NULL);
  // initialize the data
  for (i = 0; i < dims[0]; ++i) {
    for (j = 0; j < dims[1]; ++j) {
      mem[i * dims[1] + j] = i + j;
    }
  }
  ck_assert_msg(H5Dwrite(dataset, H5TYPE, H5S_ALL, H5S_ALL, 
      H5P_DEFAULT, mem) >=0, NULL);
  ck_assert_msg(h5close(dataset) >= 0, NULL);
} 
END_TEST

START_TEST(test_change_multiple_lines) {
  hid_t dataset;
  Dtype mem_input[chn * dims[1]], mem_output[chn * dims[1]];
  hsize_t offset[ndim], count[ndim];
  int i, j;

  dataset = h5open(H5F_NAME, DATASET_NAME, ndim, dims);
  // prepare data for change
  for (i = 0; i < chn; ++i) {
    for (j = 0; j < dims[1]; ++j) {
      mem_output[i * dims[1] + j] = -(i + j);
    }
  }
  // change the corresponding lines
  for (i = 0; i < chn; ++i) {
    offset[0] = chl[i];
    count[0] = 1;
    for (j = 1; j < ndim; ++j) {
      offset[j] = 0;
      count[j] = dims[j];
    }
    h5write(dataset, ndim, offset, count, mem_output + i * dims[1]);
  }
  // read those changed lines again
  for (i = 0; i < chn; ++i) {
    offset[0] = chl[i];
    count[0] = 1;
    for (j = 1; j < ndim; ++j) {
      offset[j] = 0;
      count[j] = dims[j];
    }
    h5read(dataset, ndim, offset, count, mem_input + i * dims[1]);
  }
  // assert input and output are same
  for (i = 0; i < chn; ++i) {
    for (j = 0; j < dims[1]; ++j) {
      ck_assert_msg(fabs(mem_input[i * dims[1] + j] - 
          mem_output[i * dims[1] + j]) < 1e-4, "%d, %d\n", i, j);
    }
  }
  ck_assert_msg(h5close(dataset) >= 0, NULL);
}
END_TEST

Suite* sync_param_suite() {
  Suite *suite = suite_create("hdf5utils");
  TCase *tcase = tcase_create("case");
  tcase_add_test(tcase, test_h5open_close);
  tcase_add_test(tcase, test_change_multiple_lines);
  suite_add_tcase(suite, tcase);
  return suite;
}

int main() {
  int number_failed;
  Suite *suite = sync_param_suite();
  SRunner *runner = srunner_create(suite);
  srunner_set_fork_status(runner, CK_NOFORK);
  srunner_run_all(runner, CK_NORMAL);
  number_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return number_failed;
}