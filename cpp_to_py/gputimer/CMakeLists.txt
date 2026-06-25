# Files
file(GLOB_RECURSE SRC_FILES_GT ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SRC_FILES_GT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)

# ---- Optional: GangSTA external STA engine (signoff-timer comparison, --signoff_timer gangsta) ----
# Links the prebuilt gangsta static library so the gputimer module can hand the placer's per-net RC to
# gangsta in-memory (no SPEF round-trip) and report WNS/TNS alongside Xplace's own timer. Off unless
# the library is found, so a checkout without gangsta builds and runs exactly as before.
# Lexically normalize (collapse ".."), NOT realpath: Xplace may live under a symlink whose target is
# outside the gangsta tree, so resolving the symlink would point at the wrong root. Collapsing ".."
# textually keeps us at <gangsta>/placement/Xplace/../.. == <gangsta>.
get_filename_component(GANGSTA_ROOT_DEFAULT "${PROJECT_SOURCE_DIR}/../.." ABSOLUTE)
set(GANGSTA_ROOT "${GANGSTA_ROOT_DEFAULT}" CACHE PATH "Path to the gangsta repo root (contains include/ and build/src/libgangsta.a)")
find_library(GANGSTA_LIB NAMES gangsta HINTS "${GANGSTA_ROOT}/build/src" "${GANGSTA_ROOT}/build")
find_library(TCL_LIBRARY NAMES tcl8.6 tcl)  # gangsta's embedded-Tcl SDC parser dependency
if(GANGSTA_LIB AND EXISTS "${GANGSTA_ROOT}/include/gangsta/gangsta.h" AND TCL_LIBRARY)
  set(XPLACE_GANGSTA_ENABLED ON)
  message(STATUS "gputimer: GangSTA signoff timer ENABLED (${GANGSTA_LIB})")
else()
  set(XPLACE_GANGSTA_ENABLED OFF)
  message(STATUS "gputimer: GangSTA signoff timer DISABLED (lib=${GANGSTA_LIB} tcl=${TCL_LIBRARY}); build gangsta to enable")
endif()

# Flute MP
set(FLUTE_MP_INCLUDE_DIR ${PATH_THIRDPARTY_ROOT}/flute_mp)

# CUDA/CPP GGR Kernel
add_library(gt STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GT}
                       ${SRC_FILES_GT_CUDA}
                       ${FLUTE_MP_INCLUDE_DIR}/flute.cpp
                       )

set_target_properties(gt PROPERTIES
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                       POSITION_INDEPENDENT_CODE ON)

target_include_directories(gt PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_MP_INCLUDE_DIR})
target_link_libraries(gt PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser OpenMP::OpenMP_CXX stdc++fs pthread)
target_compile_options(gt PRIVATE -fPIC)
target_compile_options(gt PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

if(XPLACE_GANGSTA_ENABLED)
  # gangsta_signoff.cpp (globbed into gt) uses only the gangsta C API header, so no C++-standard
  # coupling. The prebuilt static archive carries Taskflow (header-only) already; Tcl is its one
  # external runtime dependency and must be linked into the final module.
  target_include_directories(gt PRIVATE ${GANGSTA_ROOT}/include)
  target_compile_definitions(gt PRIVATE XPLACE_HAVE_GANGSTA=1)
  target_link_libraries(gt PRIVATE ${GANGSTA_LIB} ${TCL_LIBRARY})
endif()

pybind11_add_module(gputimer MODULE PyBindCppMain.cpp)

target_include_directories(gputimer PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_MP_INCLUDE_DIR})
target_link_libraries(gputimer PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser gt)
target_compile_definitions(gputimer PRIVATE
                            TORCH_EXTENSION_NAME=gputimer
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})
if(XPLACE_GANGSTA_ENABLED)
  # gt links gangsta PRIVATEly; re-link here so the archive's symbols resolve in the final module.
  target_include_directories(gputimer PRIVATE ${GANGSTA_ROOT}/include)
  target_compile_definitions(gputimer PRIVATE XPLACE_HAVE_GANGSTA=1)
  target_link_libraries(gputimer PRIVATE ${GANGSTA_LIB} ${TCL_LIBRARY})
endif()

install(TARGETS gputimer DESTINATION ${XPLACE_LIB_DIR})