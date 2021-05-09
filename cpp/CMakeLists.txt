cmake_minimum_required(VERSION 3.2)
project(superpoint)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_compile_options(-Wall -Wextra -Werror)

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
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} )
