cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolov4)


find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include_directories(${PROJECT_SOURCE_DIR}/src)

FILE( GLOB source_src
  src/main.cc
  src/darknet.cc
  src/config.cc
)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(yolov4 ${source_src})
target_link_libraries(yolov4 ${OpenCV_LIBS} ${TORCH_LIBRARIES} )
set_property(TARGET yolov4 PROPERTY CXX_STANDARD 14)
