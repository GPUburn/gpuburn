cmake_minimum_required(VERSION 2.6)
project(StressTest)
find_package(CUDA) 
#set(CMAKE_INCLUDE_PATH "../../../common/inc/")
set(INSTALL_PATH "./")
include_directories("../../../common/inc/")

# set source files
set(PROJECT_SRC main.cpp stress_test.cu)
set(PROJECT_HEADER stress_test.h)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
#set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

# debug options
# set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")

cuda_add_executable(stresstest ${PROJECT_SRC} OPTIONS ${GENCODE})
install(TARGETS stresstest 
        RUNTIME DESTINATION ${INSTALL_PATH})
