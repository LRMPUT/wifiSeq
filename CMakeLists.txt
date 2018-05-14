cmake_minimum_required(VERSION 3.5)
project(wifiSeq)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

# Boost
find_package(Boost COMPONENTS system filesystem serialization)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

set(wifiSeq_SRC
        src/main.cpp
        src/LocationWiFi.cpp)

add_executable(wifiSeq ${wifiSeq_SRC})

target_link_libraries(wifiSeq
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS})