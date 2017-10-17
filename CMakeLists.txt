CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project
PROJECT(FP16 LANGUAGES C CXX)

# ---[ Options.
OPTION(FP16_BUILD_TESTS "Build FP16 unit tests" ON)
OPTION(FP16_BUILD_BENCHMARKS "Build FP16 micro-benchmarks" ON)

# ---[ CMake options
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
IF(FP16_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Build flags
IF(NOT MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c99")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

# ---[ FP16 library
ADD_LIBRARY(fp16 INTERFACE)
TARGET_INCLUDE_DIRECTORIES(fp16 INTERFACE include)

INSTALL(FILES include/fp16.h
    include/fp16/bitcasts.h
    include/fp16/fp16.h
    include/fp16/psimd.h
    include/fp16/__init__.py
    include/fp16/avx.py
    include/fp16/avx2.py
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

IF(FP16_BUILD_TESTS OR FP16_BUILD_BENCHMARKS)
  FIND_PACKAGE(PSIMD REQUIRED)
ENDIF()

IF(FP16_BUILD_TESTS)
  FIND_PACKAGE(GTest REQUIRED)

  ADD_EXECUTABLE(ieee-to-fp32-bits-test test/ieee-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-bits-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(ieee-to-fp32-bits-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(ieee-to-fp32-bits ieee-to-fp32-bits-test)

  ADD_EXECUTABLE(ieee-to-fp32-value-test test/ieee-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-value-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(ieee-to-fp32-value-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(ieee-to-fp32-value ieee-to-fp32-value-test)

  ADD_EXECUTABLE(ieee-from-fp32-value-test test/ieee-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-from-fp32-value-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(ieee-from-fp32-value-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(ieee-from-fp32-value ieee-from-fp32-value-test)

  ADD_EXECUTABLE(alt-to-fp32-bits-test test/alt-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-bits-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(alt-to-fp32-bits-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(alt-to-fp32-bits alt-to-fp32-bits-test)

  ADD_EXECUTABLE(alt-to-fp32-value-test test/alt-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-value-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(alt-to-fp32-value-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(alt-to-fp32-value alt-to-fp32-value-test)

  ADD_EXECUTABLE(alt-from-fp32-value-test test/alt-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-from-fp32-value-test PRIVATE test ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(alt-from-fp32-value-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(alt-from-fp32-value alt-from-fp32-value-test)

  ADD_EXECUTABLE(bitcasts-test test/bitcasts.cc)
  TARGET_INCLUDE_DIRECTORIES(bitcasts-test PRIVATE ${GTEST_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(bitcasts-test fp16 ${GTEST_BOTH_LIBRARIES})
  ADD_TEST(bitcasts bitcasts-test)
ENDIF()

IF(FP16_BUILD_BENCHMARKS)
  FIND_PACKAGE(GBenchmark REQUIRED)

  ADD_EXECUTABLE(ieee-element-bench bench/ieee-element.cc)
  ADD_DEPENDENCIES(ieee-element-bench ${PSIMD_DEPENDENCY})
  TARGET_INCLUDE_DIRECTORIES(ieee-element-bench PRIVATE ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(ieee-element-bench fp16 ${GBENCHMARK_LIBRARIES})

  ADD_EXECUTABLE(alt-element-bench bench/alt-element.cc)
  ADD_DEPENDENCIES(alt-element-bench ${PSIMD_DEPENDENCY})
  TARGET_INCLUDE_DIRECTORIES(alt-element-bench PRIVATE ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(alt-element-bench fp16 ${GBENCHMARK_LIBRARIES})

  ADD_EXECUTABLE(from-ieee-array-bench bench/from-ieee-array.cc)
  ADD_DEPENDENCIES(from-ieee-array-bench ${PSIMD_DEPENDENCY})
  TARGET_COMPILE_DEFINITIONS(from-ieee-array-bench PRIVATE "-DFP16_COMPARATIVE_BENCHMARKS=1")
  TARGET_INCLUDE_DIRECTORIES(from-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR} ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(from-ieee-array-bench fp16 ${GBENCHMARK_LIBRARIES})

  ADD_EXECUTABLE(from-alt-array-bench bench/from-alt-array.cc)
  ADD_DEPENDENCIES(from-alt-array-bench ${PSIMD_DEPENDENCY})
  TARGET_COMPILE_DEFINITIONS(from-alt-array-bench PRIVATE "-DFP16_COMPARATIVE_BENCHMARKS=1")
  TARGET_INCLUDE_DIRECTORIES(from-alt-array-bench PRIVATE ${PROJECT_SOURCE_DIR} ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(from-alt-array-bench fp16 ${GBENCHMARK_LIBRARIES})

  ADD_EXECUTABLE(to-ieee-array-bench bench/to-ieee-array.cc)
  ADD_DEPENDENCIES(to-ieee-array-bench ${PSIMD_DEPENDENCY})
  TARGET_COMPILE_DEFINITIONS(to-ieee-array-bench PRIVATE "-DFP16_COMPARATIVE_BENCHMARKS=1")
  TARGET_INCLUDE_DIRECTORIES(to-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR} ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(to-ieee-array-bench fp16 ${GBENCHMARK_LIBRARIES})

  ADD_EXECUTABLE(to-alt-array-bench bench/to-alt-array.cc)
  ADD_DEPENDENCIES(to-alt-array-bench ${PSIMD_DEPENDENCY})
  TARGET_COMPILE_DEFINITIONS(to-alt-array-bench PRIVATE "-DFP16_COMPARATIVE_BENCHMARKS=1")
  TARGET_INCLUDE_DIRECTORIES(to-alt-array-bench PRIVATE ${PROJECT_SOURCE_DIR} ${PSIMD_INCLUDE_DIRS} ${GBENCHMARK_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(to-alt-array-bench fp16 ${GBENCHMARK_LIBRARIES})
ENDIF()
