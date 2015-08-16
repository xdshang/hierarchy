#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <check.h>

#include "sync_param.h"

#define H5FILE_NAME "sync_testdata.h5"
#define NUM 20
#define DIM 20

#define SIZE1 10
#define SIZE2 12
#define SIZE3 1
#define SIZE4 0
#define REF_SIZE1 7
#define REF_SIZE2 10
#define REF_SIZE3 1
#define REF_SIZE4 0
#define REF_RM_SIZE2 3
#define REF_RM_SIZE3 10
#define REF_RM_SIZE4 1

int list1[SIZE1] = {2, 5, 5, 3, 1, 4, 11, 9, 9, 9};
int list2[SIZE2] = {19, 2, 1, 1, 1, 3, 18, 12, 13, 4, 7, 10};
int list3[SIZE3] = {9};
int list4[SIZE4];

int compare(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}

START_TEST(test_create_destroy) {
  int hid;
  
  hid = create_sync_param(H5FILE_NAME, NUM, DIM);
  ck_assert_msg(hid == MAX_NUM_HANDLE - 1, NULL);
  hid = create_sync_param(H5FILE_NAME, NUM, DIM);
  ck_assert_msg(hid == MAX_NUM_HANDLE - 2, NULL);

  ck_assert_msg(destroy_sync_param(MAX_NUM_HANDLE - 1) == 0, NULL);
  ck_assert_msg(destroy_sync_param(MAX_NUM_HANDLE - 2) == 0, NULL);

  ck_assert_msg(destroy_sync_param(0) == -2, NULL);
} 
END_TEST

START_TEST(test_list_management) {
  int ref_list1[REF_SIZE1] = {1, 2, 3, 4, 5, 9, 11};
  int ref_list2[REF_SIZE2] = {1, 2, 3, 4, 7, 10, 12, 13, 18, 19};
  int ref_list3[REF_SIZE3] = {9};
  int ref_list4[REF_SIZE4];
  int ref_rm_list2[REF_RM_SIZE2] = {5, 9, 11};
  int ref_rm_list3[REF_RM_SIZE3] = {1, 2, 3, 4, 7, 10, 12, 13, 18, 19};
  int ref_rm_list4[REF_RM_SIZE4] = {9};
  int list[MAX_NUM_LIST];
  int hid, i;
  PrefetchArgs args;
  pthread_t pt;

  hid = create_sync_param(H5FILE_NAME, NUM, DIM);
  args.hid = hid;

  args.size = SIZE1;
  args.list = list1;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  ck_assert_msg(get_curr_size(hid) == REF_SIZE1, NULL);
  memcpy(list, get_curr_list(hid), REF_SIZE1 * sizeof(int));
  qsort(list, REF_SIZE1, sizeof(int), compare);
  for (i = 0; i < REF_SIZE1; ++i) {
    ck_assert_msg(list[i] == ref_list1[i], "%d\n", i);
  }
  ck_assert_msg(get_rm_size(hid) == 0, NULL);

  args.size = SIZE2;
  args.list = list2;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  ck_assert_msg(get_curr_size(hid) == REF_SIZE2, NULL);
  memcpy(list, get_curr_list(hid), REF_SIZE2 * sizeof(int));
  qsort(list, REF_SIZE2, sizeof(int), compare);
  for (i = 0; i < REF_SIZE2; ++i) {
    ck_assert_msg(list[i] == ref_list2[i], "%d\n", i);
  }
  ck_assert_msg(get_rm_size(hid) == REF_RM_SIZE2, NULL);
  memcpy(list, get_rm_list(hid), REF_RM_SIZE2 * sizeof(int));
  qsort(list, REF_RM_SIZE2, sizeof(int), compare);
  for (i = 0; i < REF_RM_SIZE2; ++i) {
    ck_assert_msg(list[i] == ref_rm_list2[i], "%d\n", i);
  }

  args.size = SIZE3;
  args.list = list3;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  ck_assert_msg(get_curr_size(hid) == REF_SIZE3, NULL);
  memcpy(list, get_curr_list(hid), REF_SIZE3 * sizeof(int));
  qsort(list, REF_SIZE3, sizeof(int), compare);
  for (i = 0; i < REF_SIZE3; ++i) {
    ck_assert_msg(list[i] == ref_list3[i], "%d\n", i);
  }
  ck_assert_msg(get_rm_size(hid) == REF_RM_SIZE3, NULL);
  memcpy(list, get_rm_list(hid), REF_RM_SIZE3 * sizeof(int));
  qsort(list, REF_RM_SIZE3, sizeof(int), compare);
  for (i = 0; i < REF_RM_SIZE3; ++i) {
    ck_assert_msg(list[i] == ref_rm_list3[i], "%d\n", i);
  }

  args.size = SIZE4;
  args.list = list4;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  ck_assert_msg(get_curr_size(hid) == REF_SIZE4, NULL);
  memcpy(list, get_curr_list(hid), REF_SIZE4 * sizeof(int));
  qsort(list, REF_SIZE4, sizeof(int), compare);
  for (i = 0; i < REF_SIZE4; ++i) {
    ck_assert_msg(list[i] == ref_list4[i], "%d\n", i);
  }
  ck_assert_msg(get_rm_size(hid) == REF_RM_SIZE4, NULL);
  memcpy(list, get_rm_list(hid), REF_RM_SIZE4 * sizeof(int));
  qsort(list, REF_RM_SIZE4, sizeof(int), compare);
  for (i = 0; i < REF_RM_SIZE4; ++i) {
    ck_assert_msg(list[i] == ref_rm_list4[i], "%d\n", i);
  }
}
END_TEST

START_TEST(test_prefetching) {
  PrefetchArgs args;
  const Dtype* cdata;
  Dtype* data;
  pthread_t pt;
  int i, j, ts, list[SIZE1 + SIZE2 + SIZE3];
  // generate unique timestamp for this test
  srand(time(NULL));
  ts = rand() % 100;

  args.hid = create_sync_param(H5FILE_NAME, NUM, DIM);
  ck_assert_msg(args.hid >=0, NULL);

  args.size = SIZE1;
  args.list = list1;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  // prefetch list2 while processing list1
  args.size = SIZE2;
  args.list = list2;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  for (i = 0; i < SIZE1; ++i) {
    data = mutable_sync_param(args.hid, list1[i]);
    for (j = 0; j < DIM; ++j) {
      data[j] = list1[i] + j + ts;
    }
  }
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  // prefetch list3 while processing list2
  args.size = SIZE3;
  args.list = list3;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  for (i = 0; i < SIZE2; ++i) {
    data = mutable_sync_param(args.hid, list2[i]);
    for (j = 0; j < DIM; ++j) {
      data[j] = list2[i] + j + ts;
    }
  }
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  // prefetch list4 while processing list3
  args.size = SIZE4;
  args.list = list4;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  for (i = 0; i < SIZE3; ++i) {
    data = mutable_sync_param(args.hid, list3[i]);
    for (j = 0; j < DIM; ++j) {
      data[j] = list3[i] + j + ts;
    }
  }
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  // destroy to write back the remaining
  ck_assert_msg(destroy_sync_param(args.hid) == 0, NULL);

  // read-in data again and check
  args.hid = create_sync_param(H5FILE_NAME, NUM, DIM);
  // append list1, list2 and list3 to list
  for (i = 0; i < SIZE1; ++i) {
    list[i] = list1[i];
  }
  for (i = SIZE1; i < (SIZE1 + SIZE2); ++i) {
    list[i] = list2[i - SIZE1];
  }
  for (i = SIZE1 + SIZE2; i < (SIZE1 + SIZE2 + SIZE3); ++i) {
    list[i] = list3[i - SIZE1 - SIZE2];
  }
  // fetch
  args.size = SIZE1 + SIZE2 + SIZE3;
  args.list = list;
  pthread_create(&pt, NULL, prefetch_sync_param, (void*)&args);
  pthread_join(pt, NULL);
  ck_assert_msg(args.status == 0, NULL);
  // check
  for (i = 0; i < SIZE1 + SIZE2 + SIZE3; ++i) {
    cdata = sync_param(args.hid, list[i]);
    for (j = 0; j < DIM; ++j) {
      ck_assert_msg(fabs(cdata[j] - (list[i] + j + ts)) 
          < 1e-4, "%d, %d\n", i, j);
    }
  }
  ck_assert_msg(destroy_sync_param(args.hid) == 0, NULL);
}
END_TEST

Suite* sync_param_suite() {
  Suite *suite = suite_create("sync_param");
  TCase *tcase = tcase_create("case");
  tcase_add_test(tcase, test_create_destroy);
  tcase_add_test(tcase, test_list_management);
  tcase_add_test(tcase, test_prefetching);
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