cmake_minimum_required(VERSION 3.19...3.23)
project(niftyfusion LANGUAGES CXX)

function(_target_cxx_std target std)
  target_compile_features(${target} PRIVATE cxx_std_${std})
  set_target_properties(
    ${target}
    PROPERTIES
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )
endfunction()

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.10.0
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cfusion cfusion.cpp)
target_include_directories(cfusion PRIVATE /home/jo/tools/anaconda3/envs/niftyfusion2/include)
_target_cxx_std(cfusion 17)

add_custom_command(
  TARGET cfusion POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:cfusion>" "fusion/cc/$<TARGET_FILE_NAME:cfusion>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
