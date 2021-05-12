cmake_minimum_required(VERSION 3.2)
project(superpoint)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

if (EXISTS "/usr/lib/aarch64-linux-gnu/tegra")
    message ("Nvidia Jetson platform recognized")
    set(TORCH_LIB_PATH  $ENV{HOME}/.local/lib/python3.6/site-packages/)
    set(Torch_DIR ${TORCH_LIB_PATH}/torch)
    set(CMAKE_PREFIX_PATH ${TORCH_LIB_PATH}/torch)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_compile_options(-Wall -Wextra -Werror)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_FILES
    src/main.cc
    src/camera.h
    src/camera.cc
    src/torchutis.h
    src/torchutis.cc
    src/settings.h
    src/settings.cc
    src/superpoint.h
    src/superpoint.cc
    src/model.h
    src/model.cc
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} )
