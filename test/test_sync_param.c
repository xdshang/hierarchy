#include <stdlib.h>
#include <check.h>

#include "sync_param.h"

START_TEST(test_creation) {
  create_sync_param("syn1.h5", 10, 20);
  ck_assert_msg(1, NULL);
} 
END_TEST

Suite* sync_param_suite() {
  Suite *suite = suite_create("sync_param");
  TCase *tcase = tcase_create("case");
  tcase_add_test(tcase, test_creation);
  suite_add_tcase(suite, tcase);
  return suite;
}

int main() {
  int number_failed;
  Suite *suite = sync_param_suite();
  SRunner *runner = srunner_create(suite);
  srunner_run_all(runner, CK_NORMAL);
  number_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return number_failed;
}