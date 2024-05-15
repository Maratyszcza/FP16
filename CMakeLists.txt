CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project
PROJECT(FP16 C CXX)

# ---[ Options.
OPTION(FP16_BUILD_TESTS "Build FP16 unit tests" ON)
OPTION(FP16_BUILD_BENCHMARKS "Build FP16 micro-benchmarks" ON)
OPTION(USE_SYSTEM_LIBS "Use system libraries instead of downloading and building them" OFF)
OPTION(USE_SYSTEM_PSIMD "Use system psimd instead of downloading and building it" ${USE_SYSTEM_LIBS})
OPTION(USE_SYSTEM_GOOGLEBENCHMARK "Use system Google Benchmark instead of downloading and building it" ${USE_SYSTEM_LIBS})
OPTION(USE_SYSTEM_GOOGLETEST "Use system Google Test instead of downloading and building it" ${USE_SYSTEM_LIBS})

# ---[ CMake options
IF(FP16_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

MACRO(FP16_TARGET_ENABLE_CXX11 target)
  SET_TARGET_PROPERTIES(${target} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS YES)
ENDMACRO()

# ---[ Download deps
IF(USE_SYSTEM_PSIMD)
  FIND_PACKAGE(psimd REQUIRED)
ELSEIF(NOT DEFINED PSIMD_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PSimd to ${CMAKE_BINARY_DIR}/psimd-source (define PSIMD_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CMAKE_BINARY_DIR}/psimd-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
  SET(PSIMD_SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source" CACHE STRING "PSimd source directory")
ENDIF()

IF(FP16_BUILD_TESTS)
  IF(USE_SYSTEM_GOOGLETEST)
    FIND_PACKAGE(GTest REQUIRED)
  ELSEIF(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Test to ${CMAKE_BINARY_DIR}/googletest-source (define GOOGLETEST_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
    SET(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source" CACHE STRING "Google Test source directory")
  ENDIF()
ENDIF()

IF(FP16_BUILD_BENCHMARKS)
  IF(USE_SYSTEM_GOOGLEBENCHMARK)
    FIND_PACKAGE(benchmark REQUIRED)
  ELSEIF(NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
  ENDIF()
ENDIF()

# ---[ FP16 library
ADD_LIBRARY(fp16 INTERFACE)
TARGET_INCLUDE_DIRECTORIES(fp16 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ---[ Configure psimd
IF(NOT TARGET psimd AND NOT USE_SYSTEM_PSIMD)
  ADD_SUBDIRECTORY(
    "${PSIMD_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/psimd")
ENDIF()

INCLUDE(CMakePackageConfigHelpers)
CONFIGURE_PACKAGE_CONFIG_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)

INSTALL(TARGETS fp16 EXPORT ${CMAKE_PROJECT_NAME}Targets)
INSTALL(EXPORT ${CMAKE_PROJECT_NAME}Targets
  FILE ${CMAKE_PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
)
INSTALL(DIRECTORY include/ DESTINATION include)

IF(FP16_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest AND NOT USE_SYSTEM_GOOGLETEST)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googletest")
  ENDIF()

  # ---[ Build FP16 unit tests
  ADD_EXECUTABLE(ieee-to-fp32-bits-test test/ieee-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-bits-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-to-fp32-bits-test fp16 gtest gtest_main)
  ADD_TEST(ieee-to-fp32-bits ieee-to-fp32-bits-test)

  ADD_EXECUTABLE(ieee-to-fp32-value-test test/ieee-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-to-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(ieee-to-fp32-value ieee-to-fp32-value-test)

  ADD_EXECUTABLE(ieee-from-fp32-value-test test/ieee-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-from-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-from-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(ieee-from-fp32-value ieee-from-fp32-value-test)

  ADD_EXECUTABLE(alt-to-fp32-bits-test test/alt-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-bits-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-to-fp32-bits-test fp16 gtest gtest_main)
  ADD_TEST(alt-to-fp32-bits alt-to-fp32-bits-test)

  ADD_EXECUTABLE(alt-to-fp32-value-test test/alt-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-to-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(alt-to-fp32-value alt-to-fp32-value-test)

  ADD_EXECUTABLE(alt-from-fp32-value-test test/alt-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-from-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-from-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(alt-from-fp32-value alt-from-fp32-value-test)

  ADD_EXECUTABLE(bitcasts-test test/bitcasts.cc)
  TARGET_LINK_LIBRARIES(bitcasts-test fp16 gtest gtest_main)
  ADD_TEST(bitcasts bitcasts-test)
ENDIF()

IF(FP16_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark AND NOT USE_SYSTEM_GOOGLEBENCHMARK)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googlebenchmark")
  ENDIF()

  # ---[ Build FP16 benchmarks
  ADD_EXECUTABLE(ieee-element-bench bench/ieee-element.cc)
  TARGET_COMPILE_DEFINITIONS(ieee-element-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(ieee-element-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(ieee-element-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(alt-element-bench bench/alt-element.cc)
  TARGET_LINK_LIBRARIES(alt-element-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(from-ieee-array-bench bench/from-ieee-array.cc)
  FP16_TARGET_ENABLE_CXX11(from-ieee-array-bench)
  TARGET_COMPILE_DEFINITIONS(from-ieee-array-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(from-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(from-ieee-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(from-alt-array-bench bench/from-alt-array.cc)
  FP16_TARGET_ENABLE_CXX11(from-alt-array-bench)
  TARGET_LINK_LIBRARIES(from-alt-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(to-ieee-array-bench bench/to-ieee-array.cc)
  FP16_TARGET_ENABLE_CXX11(to-ieee-array-bench)
  TARGET_COMPILE_DEFINITIONS(to-ieee-array-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(to-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(to-ieee-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(to-alt-array-bench bench/to-alt-array.cc)
  FP16_TARGET_ENABLE_CXX11(to-alt-array-bench)
  TARGET_LINK_LIBRARIES(to-alt-array-bench fp16 psimd benchmark)
ENDIF()
