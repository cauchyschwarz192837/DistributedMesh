cmake_minimum_required(VERSION 3.20)

project(CMakeProject1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(MPI REQUIRED)

set(SLANGC_EXECUTABLE "$ENV{VULKAN_SDK}/Bin/slangc.exe")

function(add_slang_shader_target TARGET)
  cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})
  set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set(ENTRY_POINTS -entry vertMain -entry fragMain)

  add_custom_command(
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )

  add_custom_command(
    OUTPUT ${SHADERS_DIR}/slang.spv
    COMMAND ${SLANGC_EXECUTABLE}
            ${SHADER_SOURCES}
            -target spirv
            -profile spirv_1_4
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            ${ENTRY_POINTS}
            -o slang.spv
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
  )

  add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_slang_shader_target(compileShaders
  SOURCES ${CMAKE_CURRENT_LIST_DIR}/shader.slang
)

add_executable(CMakeProject1
  CMakeProject1.cpp
  comm.cpp
)

target_link_libraries(CMakeProject1
  PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    MPI::MPI_CXX
)

add_dependencies(CMakeProject1 compileShaders)

target_compile_definitions(CMakeProject1 PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)