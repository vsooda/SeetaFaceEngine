cmake_minimum_required(VERSION 2.8)
project(FaceDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCE_FILES
    src/classifier/lab_boosted_classifier.cpp
    src/classifier/mlp.cpp
    src/classifier/surf_mlp.cpp
    src/feat/lab_feature_map.cpp
    src/feat/surf_feature_map.cpp
    src/io/lab_boost_model_reader.cpp
    src/io/surf_mlp_model_reader.cpp
    src/test/facedetection_test.cpp
    src/util/image_pyramid.cpp
    src/util/nms.cpp
    src/face_detection.cpp
    src/fust.cpp)

add_executable(FaceDetection ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(FaceDetection ${OpenCV_LIBS} )