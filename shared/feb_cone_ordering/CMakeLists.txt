cmake_minimum_required(VERSION 3.8)
project(feb_cone_ordering)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "srv/OrderCones.srv" DEPENDENCIES geometry_msgs)

add_executable(cone_ordering_node src/cone_ordering_node.cpp)
target_include_directories(cone_ordering_node PRIVATE src)
ament_target_dependencies(cone_ordering_node rclcpp geometry_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(cone_ordering_node "${cpp_typesupport_target}")

add_executable(cone_order_cli src/cone_order_cli.cpp)
target_include_directories(cone_order_cli PRIVATE src)

install(TARGETS cone_ordering_node cone_order_cli DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
