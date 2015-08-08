FIND_PACKAGE(Check REQUIRED)

INCLUDE_DIRECTORIES(${CHECK_INCLUDE_DIRS})

SET(LIBS ${CHECK_LIBRARIES} hs)

FILE(GLOB TEST_SOURCES *.c)

FOREACH(test_src ${TEST_SOURCES})

  GET_FILENAME_COMPONENT(test_target ${test_src} NAME_WE)

  ADD_EXECUTABLE(${test_target} ${test_src})

  TARGET_LINK_LIBRARIES(${test_target} ${LIBS})

  ADD_TEST(${test_target} ${test_target})

ENDFOREACH(test_src ${TEST_SOURCES})