# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/haihe/tmp/cmake/cmake-3.17.0-rc3/bin/cmake

# The command to remove a file.
RM = /home/haihe/tmp/cmake/cmake-3.17.0-rc3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/haihe/RapidMatch-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haihe/RapidMatch-main/build

# Include any dependencies generated for this target.
include preprocessing/CMakeFiles/GraphConverter.out.dir/depend.make

# Include the progress variables for this target.
include preprocessing/CMakeFiles/GraphConverter.out.dir/progress.make

# Include the compile flags for this target's objects.
include preprocessing/CMakeFiles/GraphConverter.out.dir/flags.make

preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o: preprocessing/CMakeFiles/GraphConverter.out.dir/flags.make
preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o: ../preprocessing/GraphConverter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/RapidMatch-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o"
	cd /home/haihe/RapidMatch-main/build/preprocessing && /opt/rh/devtoolset-8/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o -c /home/haihe/RapidMatch-main/preprocessing/GraphConverter.cpp

preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.i"
	cd /home/haihe/RapidMatch-main/build/preprocessing && /opt/rh/devtoolset-8/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/RapidMatch-main/preprocessing/GraphConverter.cpp > CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.i

preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.s"
	cd /home/haihe/RapidMatch-main/build/preprocessing && /opt/rh/devtoolset-8/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/RapidMatch-main/preprocessing/GraphConverter.cpp -o CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.s

# Object files for target GraphConverter.out
GraphConverter_out_OBJECTS = \
"CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o"

# External object files for target GraphConverter.out
GraphConverter_out_EXTERNAL_OBJECTS =

preprocessing/GraphConverter.out: preprocessing/CMakeFiles/GraphConverter.out.dir/GraphConverter.cpp.o
preprocessing/GraphConverter.out: preprocessing/CMakeFiles/GraphConverter.out.dir/build.make
preprocessing/GraphConverter.out: graph/libgraph.so
preprocessing/GraphConverter.out: utility/libutility.so
preprocessing/GraphConverter.out: preprocessing/CMakeFiles/GraphConverter.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haihe/RapidMatch-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GraphConverter.out"
	cd /home/haihe/RapidMatch-main/build/preprocessing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GraphConverter.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
preprocessing/CMakeFiles/GraphConverter.out.dir/build: preprocessing/GraphConverter.out

.PHONY : preprocessing/CMakeFiles/GraphConverter.out.dir/build

preprocessing/CMakeFiles/GraphConverter.out.dir/clean:
	cd /home/haihe/RapidMatch-main/build/preprocessing && $(CMAKE_COMMAND) -P CMakeFiles/GraphConverter.out.dir/cmake_clean.cmake
.PHONY : preprocessing/CMakeFiles/GraphConverter.out.dir/clean

preprocessing/CMakeFiles/GraphConverter.out.dir/depend:
	cd /home/haihe/RapidMatch-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haihe/RapidMatch-main /home/haihe/RapidMatch-main/preprocessing /home/haihe/RapidMatch-main/build /home/haihe/RapidMatch-main/build/preprocessing /home/haihe/RapidMatch-main/build/preprocessing/CMakeFiles/GraphConverter.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : preprocessing/CMakeFiles/GraphConverter.out.dir/depend
