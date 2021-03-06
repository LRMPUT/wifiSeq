cmake_minimum_required(VERSION 3.5)
project(wifiSeq)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(include)

# Boost
find_package(Boost COMPONENTS system filesystem serialization)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

#TinyXML
set(TinyXML_LIBRARIES tinyxml)

#LBFGS
set(LBFGS_LIBRARIES lbfgs)

set(wifiSeq_SRC
        src/main.cpp
        src/Utils.cpp
        src/LocationWifi.cpp
        src/WiFiSeqFeatures.cpp
        src/Stepometer.cpp
        src/Graph.cpp
        src/fastable/FastABLE.cpp
        src/ldb/ldb.cpp
        src/pgm/Inference.cpp
        src/pgm/ParamEst.cpp
        src/pgm/ParamEstSGD.cpp
        src/pgm/Pgm.cpp
        src/pgm/UnionFind.cpp)

add_executable(wifiSeq ${wifiSeq_SRC})

target_link_libraries(wifiSeq
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${TinyXML_LIBRARIES}
                      ${LBFGS_LIBRARIES})