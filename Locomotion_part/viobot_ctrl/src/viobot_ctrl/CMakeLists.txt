cmake_minimum_required(VERSION 3.8)
project(viobot_ctrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(system_ctrl REQUIRED)
find_package(glog REQUIRED)

add_executable(algo_ctrl_node src/viobot_algo_ctrl_node.cpp)
target_include_directories(algo_ctrl_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(algo_ctrl_node glog::glog)
ament_target_dependencies(algo_ctrl_node rclcpp std_msgs system_ctrl)

install(TARGETS
  algo_ctrl_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
