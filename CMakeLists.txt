cmake_minimum_required (VERSION 3.8)
project ("obs-mdct")

if (OBSSourcePath OR DEFINED ENV{OBSSourcePath})
    # Set already
else()
    set(OBSSourcePath "C:/git/OBS/obs-studio/libobs/" CACHE PATH "Path to OBS source code (e.g., C:/Dev/obs-studio/libobs/)")
    message(SEND_ERROR "OBSSourcePath is missing. Please set this variable to the location of the OBS source (e.g., C:/Dev/obs-studio/libobs/).")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "/MD /O3 /Ob2 /DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "/MD /O3 /Ob2 /DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/RUNTIME CACHE STRING "" FORCE)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_lib_suffix 64)
else()
    message(SEND_ERROR "64-bit only!")
endif()

find_path(OBS_LIB_DIR
    NAMES obs.dll obs.lib
    HINTS
        ${OBSSourcePath}/../build/libobs/Release
        ${OBSSourcePath}/../build/libobs/Debug
    PATHS
        /usr/lib /usr/local/lib /opt/local/lib /sw/lib)

if(!OBS_LIB_DIR)
    message(SEND_ERROR "Couldn't find obs.dll or obs.lib")
endif()

find_path(OBS_PLUGINS
    NAMES obs-filters.dll
    HINTS
        ${OBSSourcePath}/../build/rundir/Release/obs-plugins/64bit
        ${OBSSourcePath}/../build/rundir/Debug/obs-plugins/64bit
        "C:/Program Files/obs-studio/obs-plugins/64bit"
    PATHS
        /usr/lib /usr/local/lib /opt/local/lib /sw/lib)

if(!OBS_PLUGINS)
    message(SEND_ERROR "Couldn't find OBS filters directory")
endif()

set(CMAKE_INSTALL_PREFIX ${OBS_PLUGINS} CACHE STRING "" FORCE)

# Source
file (GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.c ${CMAKE_SOURCE_DIR}/src/*.cpp)
file (GLOB HEADERS ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.hpp)

include_directories (include ${CMAKE_BINARY_DIR}/config)
add_library (${PROJECT_NAME} SHARED
    ${SOURCES}
    ${HEADERS}
)

# libobs
include_directories(${OBSSourcePath})
add_library (libobs SHARED IMPORTED)
set_property (TARGET libobs PROPERTY IMPORTED_LOCATION ${OBS_LIB_DIR}/obs.dll)
set_property (TARGET libobs PROPERTY IMPORTED_IMPLIB ${OBS_LIB_DIR}/obs.lib)
target_link_libraries (${PROJECT_NAME} libobs)



install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${OBS_PLUGINS})

#install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION obs-plugins/${_lib_suffix}bit)
#install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${PROJECT_NAME}.pdb DESTINATION obs-plugins/${_lib_suffix}bit CONFIGURATIONS Debug)
#install(FILES ${CMAKE_BINARY_DIR}/RUNTIME/${PROJECT_NAME}.pdb DESTINATION obs-plugins/${_lib_suffix}bit CONFIGURATIONS Debug)
#install(DIRECTORY data/ DESTINATION data/obs-plugins/${PROJECT_NAME}/)
