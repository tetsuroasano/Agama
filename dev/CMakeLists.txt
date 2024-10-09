cmake_minimum_required(VERSION 3.10)

project(Agama_test VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -O2 -march=native -std=c++11 -Wno-missing-field-initializers -Wno-cast-function-type")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/../src)

link_directories(${CMAKE_SOURCE_DIR}/../extras/lib )
link_libraries(${CMAKE_SOURCE_DIR}/../agama.a libgsl.a libgslcblas.a libunsio.a libnemo.a)

add_executable(orbit_int orbit_int.cpp)

target_link_libraries(orbit_int ${OpenMP_CXX_LIBRARIES})
