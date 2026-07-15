cmake_minimum_required(VERSION 3.20)
project(occt_cleaner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ACAD_FETCH_OCCT "Download and build OCCT V8_0_0 as an ExternalProject" OFF)
set(ACAD_OCCT_ROOT
    ""
    CACHE PATH "Root of a local OCCT package, for example C:/tools/OpenCascade")
set(ACAD_OCCT_URL
    "https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V8_0_0.zip"
    CACHE STRING "OCCT source archive URL")
set(ACAD_OCCT_INSTALL_DIR
    "${CMAKE_BINARY_DIR}/_deps/occt-install"
    CACHE PATH "Install prefix for fetched OCCT")

if(NOT ACAD_OCCT_ROOT AND NOT ACAD_FETCH_OCCT)
  set(_ACAD_OPENCASCADE_DIR_CANDIDATES)
  if(DEFINED OpenCASCADE_DIR)
    list(APPEND _ACAD_OPENCASCADE_DIR_CANDIDATES "${OpenCASCADE_DIR}")
  endif()
  if(DEFINED ENV{OpenCASCADE_DIR})
    list(APPEND _ACAD_OPENCASCADE_DIR_CANDIDATES "$ENV{OpenCASCADE_DIR}")
  endif()

  foreach(_ACAD_OPENCASCADE_DIR IN LISTS _ACAD_OPENCASCADE_DIR_CANDIDATES)
    if(EXISTS "${_ACAD_OPENCASCADE_DIR}/OpenCASCADEConfig.cmake")
      get_filename_component(_ACAD_MAYBE_OCCT_ROOT "${_ACAD_OPENCASCADE_DIR}/.." ABSOLUTE)
      if(EXISTS "${_ACAD_MAYBE_OCCT_ROOT}/inc/Standard.hxx"
         AND EXISTS "${_ACAD_MAYBE_OCCT_ROOT}/win64")
        set(ACAD_OCCT_ROOT
            "${_ACAD_MAYBE_OCCT_ROOT}"
            CACHE PATH "Root of a local OCCT package" FORCE)
        message(STATUS "Using ACAD_OCCT_ROOT inferred from OpenCASCADE_DIR: ${ACAD_OCCT_ROOT}")
        break()
      endif()
    endif()
  endforeach()
endif()

# Minimal OCCT libraries used by the STEP -> BRep graph cleaner. Keeping this
# list explicit lets the fetched-OCCT path link without requiring
# OpenCASCADEConfig.cmake during the first configure pass.
set(ACAD_OCCT_LIBS
    TKernel
    TKMath
    TKG2d
    TKG3d
    TKGeomBase
    TKGeomAlgo
    TKBRep
    TKTopAlgo
    TKPrim
    TKBO
    TKXSBase
    TKDE
    TKDESTEP)

if(ACAD_FETCH_OCCT)
  include(ExternalProject)

  ExternalProject_Add(occt_ep
      URL "${ACAD_OCCT_URL}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      PREFIX "${CMAKE_BINARY_DIR}/_deps/occt"
      CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX=${ACAD_OCCT_INSTALL_DIR}"
        "-DBUILD_MODULE_Draw=OFF"
        "-DBUILD_MODULE_Visualization=OFF"
        "-DBUILD_DOC_Overview=OFF"
        "-DBUILD_SAMPLES_QT=OFF"
        "-DBUILD_Inspector=OFF"
        "-DUSE_TCL=OFF"
        "-DUSE_TK=OFF"
        "-DUSE_FREETYPE=OFF"
        "-DUSE_FREEIMAGE=OFF"
        "-DUSE_OPENGL=OFF"
        "-DUSE_VTK=OFF"
        "-DUSE_TBB=OFF"
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
      INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config $<CONFIG>)

  set(_OCCT_INCLUDE_DIRS
      "${ACAD_OCCT_INSTALL_DIR}/include"
      "${ACAD_OCCT_INSTALL_DIR}/include/opencascade")
  set(_OCCT_LINK_DIRS
      "${ACAD_OCCT_INSTALL_DIR}/lib"
      "${ACAD_OCCT_INSTALL_DIR}/bin"
      "${ACAD_OCCT_INSTALL_DIR}/win64/vc14/lib"
      "${ACAD_OCCT_INSTALL_DIR}/win64/vc15/lib"
      "${ACAD_OCCT_INSTALL_DIR}/win64/vc16/lib"
      "${ACAD_OCCT_INSTALL_DIR}/win64/vc17/lib")
  set(_OCCT_LIBS ${ACAD_OCCT_LIBS})
elseif(ACAD_OCCT_ROOT)
  set(_OCCT_INCLUDE_DIRS
      "${ACAD_OCCT_ROOT}/inc"
      "${ACAD_OCCT_ROOT}/include"
      "${ACAD_OCCT_ROOT}/include/opencascade")
  set(_OCCT_LINK_DIRS
      "${ACAD_OCCT_ROOT}/win64/vc14/lib"
      "${ACAD_OCCT_ROOT}/win64/vc15/lib"
      "${ACAD_OCCT_ROOT}/win64/vc16/lib"
      "${ACAD_OCCT_ROOT}/win64/vc17/lib"
      "${ACAD_OCCT_ROOT}/lib"
      "${ACAD_OCCT_ROOT}/bin")
  set(_OCCT_LIBS ${ACAD_OCCT_LIBS})
else()
  find_package(OpenCASCADE REQUIRED)
  set(_OCCT_INCLUDE_DIRS ${OpenCASCADE_INCLUDE_DIR} ${OpenCASCADE_INCLUDE_DIRS})
  set(_OCCT_LINK_DIRS)
  # This tool is a headless STEP -> BRep-graph converter and never touches
  # OCCT's DRAW/visualization toolkits. Distro OCCT packages (e.g. Debian/
  # Ubuntu libocct-*) report those toolkits in OpenCASCADE_LIBRARIES, which
  # in turn pull in Tk/Tcl/OpenGL dev libraries. Systems without those dev
  # packages installed (common on minimal CI/build images) then fail to
  # link this tool for no reason related to its actual dependencies, so
  # they're filtered out here.
  set(_OCCT_LIBS ${OpenCASCADE_LIBRARIES})
  list(FILTER _OCCT_LIBS EXCLUDE REGEX
       "Draw|OpenGl|V3d|Service|MeshVS|ViewerTest|QADraw|TopTest|XDEDRAW|XSDRAW|TObjDRAW|DCAF")
endif()

add_executable(occt_cleaner src/main.cpp)

if(ACAD_FETCH_OCCT)
  add_dependencies(occt_cleaner occt_ep)
endif()

target_include_directories(occt_cleaner PRIVATE ${_OCCT_INCLUDE_DIRS})
target_link_directories(occt_cleaner PRIVATE ${_OCCT_LINK_DIRS})
target_link_libraries(occt_cleaner PRIVATE ${_OCCT_LIBS})

if(MSVC)
  target_compile_options(occt_cleaner PRIVATE /W4 /permissive-)
else()
  target_compile_options(occt_cleaner PRIVATE -Wall -Wextra -Wpedantic)
endif()
