#include <stdlib.h>
#include <string.h>
#include <check.h>

#include "sync_param.h"

int compare(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}

START_TEST(test_create_destroy) {
  int hid;
  
  hid = create_sync_param("syn1.h5", 10, 20);
  ck_assert_msg(hid == MAX_NUM_HANDLE - 1, NULL);
  hid = create_sync_param("syn2.h5", 20, 10);
  ck_assert_msg(hid == MAX_NUM_HANDLE - 2, NULL);

  ck_assert_msg(destroy_sync_param(MAX_NUM_HANDLE - 1) == 0, NULL);
  hid = create_sync_param("syn3.h5", 10, 10);
  ck_assert_msg(hid == MAX_NUM_HANDLE - 1, NULL);

  ck_assert_msg(destroy_sync_param(MAX_NUM_HANDLE - 2) == 0, NULL);
  ck_assert_msg(destroy_sync_param(0) == -2, NULL);
} 
END_TEST

START_TEST(test_list_management) {
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
  int ref_list1[REF_SIZE1] = {1, 2, 3, 4, 5, 9, 11};
  int ref_list2[REF_SIZE2] = {1, 2, 3, 4, 7, 10, 12, 13, 18, 19};
  int ref_list3[REF_SIZE3] = {9};
  int ref_list4[REF_SIZE4];
  int ref_rm_list2[REF_RM_SIZE2] = {5, 9, 11};
  int ref_rm_list3[REF_RM_SIZE3] = {1, 2, 3, 4, 7, 10, 12, 13, 18, 19};
  int ref_rm_list4[REF_RM_SIZE4] = {9};
  int list[MAX_NUM_LIST];
  int hid, i;
  const int num = 20, dim = 20;

  hid = create_sync_param("syn1.h5", num, dim);

  ck_assert_msg(prefetch_sync_param(hid, SIZE1, list1) == 0, NULL);
  ck_assert_msg(get_curr_size(hid) == REF_SIZE1, NULL);
  memcpy(list, get_curr_list(hid), REF_SIZE1 * sizeof(int));
  qsort(list, REF_SIZE1, sizeof(int), compare);
  for (i = 0; i < REF_SIZE1; ++i) {
    ck_assert_msg(list[i] == ref_list1[i], "%d\n", i);
  }
  ck_assert_msg(get_rm_size(hid) == 0, NULL);

  ck_assert_msg(prefetch_sync_param(hid, SIZE2, list2) == 0, NULL);
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

  ck_assert_msg(prefetch_sync_param(hid, SIZE3, list3) == 0, NULL);
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

  ck_assert_msg(prefetch_sync_param(hid, SIZE4, list4) == 0, NULL);
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
#undef SIZE1
#undef SIZE2
#undef SIZE3
#undef SIZE4
#undef REF_SIZE1
#undef REF_SIZE2
#undef REF_SIZE3
#undef REF_SIZE4
#undef REF_RM_SIZE2
#undef REF_RM_SIZE3
#undef REF_RM_SIZR4
}
END_TEST

Suite* sync_param_suite() {
  Suite *suite = suite_create("sync_param");
  TCase *tcase = tcase_create("case");
  tcase_add_test(tcase, test_create_destroy);
  tcase_add_test(tcase, test_list_management);
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