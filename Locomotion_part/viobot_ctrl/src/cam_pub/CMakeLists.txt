cmake_minimum_required(VERSION 3.8)
project(cam_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(cam_pub src/cam_pub.cpp)

target_include_directories(cam_pub PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(cam_pub
  ${OpenCV_LIBS})

ament_target_dependencies(cam_pub
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge)

install(TARGETS
  cam_pub
  DESTINATION lib/${PROJECT_NAME})

ament_package()
