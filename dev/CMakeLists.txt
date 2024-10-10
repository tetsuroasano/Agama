# CMakeの最低バージョンを設定
cmake_minimum_required(VERSION 3.10)

# プロジェクト名とバージョン、使用する言語を指定
project(Agama_test VERSION 1.0 LANGUAGES CXX)

# コンパイルコマンドをエクスポートするオプションを有効化
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# コンパイラオプションをリスト形式で設定
set(CXX_FLAGS -fPIC -Wall -O2 -march=native -std=c++11 -Wno-missing-field-initializers -Wno-cast-function-type)

# OpenMPを探す
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    # OpenMP用のコンパイルオプションをリストに追加
    list(APPEND CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

# GSLを探す
find_package(GSL)
if(GSL_FOUND)
    # GSLのインクルードディレクトリとライブラリを設定
    set(GSL_LIBS GSL::gsl GSL::gslcblas)
else()
    # GSLが見つからない場合、手動でライブラリを設定
    set(GSL_LIBS ${CMAKE_SOURCE_DIR}/../extras/lib/libgsl.a ${CMAKE_SOURCE_DIR}/../extras/lib/libgslcblas.a)
endif()

# 共通のインクルードディレクトリを設定
set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/../src)

# 共通のリンクディレクトリを設定
link_directories(${CMAKE_SOURCE_DIR}/../extras/lib)

# ターゲット orbit_int を作成し、コンパイルオプションやインクルードディレクトリ、ライブラリを設定
add_executable(orbit_int orbit_int.cpp)
target_include_directories(orbit_int PRIVATE ${COMMON_INCLUDES})
target_compile_options(orbit_int PRIVATE ${CXX_FLAGS})
target_link_libraries(orbit_int PRIVATE ${GSL_LIBS} OpenMP::OpenMP_CXX ${CMAKE_SOURCE_DIR}/../agama.a libunsio.a libnemo.a)

# ターゲット test を作成し、コンパイルオプションやインクルードディレクトリ、ライブラリを設定
add_executable(test test.cpp)
target_include_directories(test PRIVATE ${COMMON_INCLUDES})
target_compile_options(test PRIVATE ${CXX_FLAGS})
target_link_libraries(test PRIVATE ${GSL_LIBS} OpenMP::OpenMP_CXX ${CMAKE_SOURCE_DIR}/../agama.a libunsio.a libnemo.a)

#cmake_minimum_required(VERSION 3.10)
#
#project(Agama_test VERSION 1.0 LANGUAGES CXX)
#
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -O2 -march=native -std=c++11 -Wno-missing-field-initializers -Wno-cast-function-type")
#
#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#	#include_directories(${OpenMP_CXX_INCLUDE_DIRS})
#	#link_libraries(${OpenMP_CXX_LIBRARIES})
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()
#
#find_package(GSL)
#if(GSL_FOUND)
#	include_directories(${GSL_INCLUDE_DIRS})
#	link_libraries(${GSL_LIBRARIES})
#else()
#	link_directories(${CMAKE_SOURCE_DIR}/../extras/lib )
#	link_libraries(libgsl.a libgslcblas.a)
#endif()
#
#include_directories(${CMAKE_SOURCE_DIR}/../src)
#
#link_directories(${CMAKE_SOURCE_DIR}/../extras/lib )
##link_libraries(${CMAKE_SOURCE_DIR}/../agama.a libgsl.a libgslcblas.a libunsio.a libnemo.a)
#link_libraries(${CMAKE_SOURCE_DIR}/../agama.a  libunsio.a libnemo.a)
#
#add_executable(orbit_int orbit_int.cpp)
#add_executable(test test.cpp)
#
##target_link_libraries(orbit_int ${OpenMP_CXX_LIBRARIES})
