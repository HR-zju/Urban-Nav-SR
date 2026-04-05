cmake_minimum_required(VERSION 3.8)
project(gnss_comm_impl)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Wall -Wextra -Wpedantic -fPIC)
endif()

# 查找 ROS 2 核心及接口依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gnss_comm REQUIRED)

# 查找第三方系统依赖
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

set(SOURCE_FILES
  src/app_node.cpp
  src/gnss_ros.cpp
  src/gnss_spp.cpp
  src/gnss_utility.cpp
  src/rinex_helper.cpp
  src/kml_exporter.cpp
)

add_executable(app_node ${SOURCE_FILES})

target_include_directories(app_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(app_node
  Eigen3::Eigen
  glog::glog
)

ament_target_dependencies(app_node
  rclcpp
  sensor_msgs
  gnss_comm
)

install(TARGETS app_node
  DESTINATION lib/${PROJECT_NAME}
)

# install(DIRECTORY include/
#   DESTINATION include/
# )

ament_package()
