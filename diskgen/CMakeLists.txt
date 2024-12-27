cmake_minimum_required(VERSION 3.10)

project(Agama_test VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# コンパイルフラグ
set(COMMON_FLAGS -fPIC -Wall -O2 -march=native -std=c++11 -Wno-missing-field-initializers -Wno-cast-function-type)
#set(COMMON_FLAGS -fPIC -O2 -std=c++11 -Nclang)

# OpenMP の設定
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
        # list(APPEND COMMON_FLAGS -fopenmp)
        list(APPEND COMMON_FLAGS ${OpenMP_CXX_FLAGS})
endif()

# GSL の設定
find_package(GSL)
if(GSL_FOUND)
	include_directories(${GSL_INCLUDE_DIRS})
	set(GSL_LIBS GSL::gsl GSL::gslcblas)
else()
	set(GSL_LIBS ${CMAKE_SOURCE_DIR}/../extras/lib/libgsl.a ${CMAKE_SOURCE_DIR}/../extras/lib/libgslcblas.a)
endif()

# 共通のヘッダディレクトリ
set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src  ${CMAKE_SOURCE_DIR}/../extras/include)

if(EXISTS ${CMAKE_SOURCE_DIR}/../extras/lib/libnemo.a)
	set(EXTRA_LIBS ${CMAKE_SOURCE_DIR}/../extras/lib/libunsio.a ${CMAKE_SOURCE_DIR}/../extras/lib/libnemo.a)
else()
	set(EXTRA_LIBS "")
endif()

# ターゲットの作成
add_executable(disk_gen main.cpp gas_scm.cpp)
target_include_directories(disk_gen PRIVATE ${COMMON_INCLUDES})
target_compile_options(disk_gen PRIVATE ${COMMON_FLAGS})
target_link_options(disk_gen PRIVATE ${COMMON_FLAGS})
#target_link_libraries(disk_gen PRIVATE ${CMAKE_SOURCE_DIR}/../agama.a GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX)
if(OpenMP_CXX_FOUND)
	target_link_libraries(disk_gen PRIVATE ${CMAKE_SOURCE_DIR}/../agama.a ${GSL_LIBS} OpenMP::OpenMP_CXX ${EXTRA_LIBS})
else()
	target_link_libraries(disk_gen PRIVATE ${CMAKE_SOURCE_DIR}/../agama.a ${GSL_LIBS}  ${EXTRA_LIBS})
endif()

### Fugaku
#cmake_minimum_required(VERSION 3.10)
#
#project(Agama_test VERSION 1.0 LANGUAGES CXX)
#
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
## コンパイルフラグ
#set(COMMON_FLAGS -fPIC -O2 -std=c++11 -Nclang)
#
## OpenMP の設定
#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#        # list(APPEND COMMON_FLAGS -fopenmp)
#        list(APPEND COMMON_FLAGS ${OpenMP_CXX_FLAGS})
#endif()
#
## GSL の設定
#find_package(GSL REQUIRED)
#if(GSL_FOUND)
#    message(STATUS "GSL found: ${GSL_LIBRARIES}")
#else()
#    message(FATAL_ERROR "GSL not found!")
#endif()
#
## 共通のヘッダディレクトリ
#set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src)
#
## ターゲットの作成
#add_executable(disk_gen main.cpp gas_scm.cpp)
#target_include_directories(disk_gen PRIVATE ${COMMON_INCLUDES})
#target_compile_options(disk_gen PRIVATE ${COMMON_FLAGS})
#target_link_options(disk_gen PRIVATE ${COMMON_FLAGS})
#target_link_libraries(disk_gen PRIVATE ${CMAKE_SOURCE_DIR}/../agama.a GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX)
