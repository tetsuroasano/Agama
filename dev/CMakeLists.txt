cmake_minimum_required(VERSION 3.10)

project(Agama_test VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CXX_FLAGS -fPIC -Wall -O2 -march=native -std=c++11 -Wno-missing-field-initializers -Wno-cast-function-type)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    list(APPEND CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

find_package(GSL REQUIRED)
if(GSL_FOUND)
	include_directories(${GSL_INCLUDE_DIRS})
    set(GSL_LIBS GSL::gsl GSL::gslcblas)
else()
    set(GSL_LIBS ${CMAKE_SOURCE_DIR}/../extras/lib/libgsl.a ${CMAKE_SOURCE_DIR}/../extras/lib/libgslcblas.a)
endif()

set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src)

link_directories(${CMAKE_SOURCE_DIR}/../extras/lib)

add_executable(orbit_int orbit_int.cpp)
target_include_directories(orbit_int PRIVATE ${COMMON_INCLUDES})
target_compile_options(orbit_int PRIVATE ${CXX_FLAGS})
target_link_libraries(orbit_int PRIVATE  ${CMAKE_SOURCE_DIR}/../agama.a libunsio.a libnemo.a ${GSL_LIBS} OpenMP::OpenMP_CXX)
# target_link_libraries(orbit_int PRIVATE ${CMAKE_SOURCE_DIR}/../agama.a ${CMAKE_SOURCE_DIR}/../extras/lib/libunsio.a ${CMAKE_SOURCE_DIR}/../extras/lib/libnemo.a ${GSL_LIBS} OpenMP::OpenMP_CXX)
