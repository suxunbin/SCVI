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
CMAKE_SOURCE_DIR = /home/haihe/common/SubgraphMatching-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haihe/common/SubgraphMatching-master/build

# Include any dependencies generated for this target.
include utility/CMakeFiles/utility.dir/depend.make

# Include the progress variables for this target.
include utility/CMakeFiles/utility.dir/progress.make

# Include the compile flags for this target's objects.
include utility/CMakeFiles/utility.dir/flags.make

utility/CMakeFiles/utility.dir/graphoperations.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/graphoperations.cpp.o: ../utility/graphoperations.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utility/CMakeFiles/utility.dir/graphoperations.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/graphoperations.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/graphoperations.cpp

utility/CMakeFiles/utility.dir/graphoperations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/graphoperations.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/graphoperations.cpp > CMakeFiles/utility.dir/graphoperations.cpp.i

utility/CMakeFiles/utility.dir/graphoperations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/graphoperations.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/graphoperations.cpp -o CMakeFiles/utility.dir/graphoperations.cpp.s

utility/CMakeFiles/utility.dir/commandparser.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/commandparser.cpp.o: ../utility/commandparser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object utility/CMakeFiles/utility.dir/commandparser.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/commandparser.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/commandparser.cpp

utility/CMakeFiles/utility.dir/commandparser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/commandparser.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/commandparser.cpp > CMakeFiles/utility.dir/commandparser.cpp.i

utility/CMakeFiles/utility.dir/commandparser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/commandparser.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/commandparser.cpp -o CMakeFiles/utility.dir/commandparser.cpp.s

utility/CMakeFiles/utility.dir/computesetintersection.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/computesetintersection.cpp.o: ../utility/computesetintersection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object utility/CMakeFiles/utility.dir/computesetintersection.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/computesetintersection.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/computesetintersection.cpp

utility/CMakeFiles/utility.dir/computesetintersection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/computesetintersection.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/computesetintersection.cpp > CMakeFiles/utility.dir/computesetintersection.cpp.i

utility/CMakeFiles/utility.dir/computesetintersection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/computesetintersection.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/computesetintersection.cpp -o CMakeFiles/utility.dir/computesetintersection.cpp.s

utility/CMakeFiles/utility.dir/bitsetoperation.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/bitsetoperation.cpp.o: ../utility/bitsetoperation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object utility/CMakeFiles/utility.dir/bitsetoperation.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/bitsetoperation.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/bitsetoperation.cpp

utility/CMakeFiles/utility.dir/bitsetoperation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/bitsetoperation.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/bitsetoperation.cpp > CMakeFiles/utility.dir/bitsetoperation.cpp.i

utility/CMakeFiles/utility.dir/bitsetoperation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/bitsetoperation.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/bitsetoperation.cpp -o CMakeFiles/utility.dir/bitsetoperation.cpp.s

utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.o: ../utility/han/intersection_algos.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/han/intersection_algos.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/han/intersection_algos.cpp

utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/han/intersection_algos.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/han/intersection_algos.cpp > CMakeFiles/utility.dir/han/intersection_algos.cpp.i

utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/han/intersection_algos.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/han/intersection_algos.cpp -o CMakeFiles/utility.dir/han/intersection_algos.cpp.s

utility/CMakeFiles/utility.dir/han/utils/util.cpp.o: utility/CMakeFiles/utility.dir/flags.make
utility/CMakeFiles/utility.dir/han/utils/util.cpp.o: ../utility/han/utils/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object utility/CMakeFiles/utility.dir/han/utils/util.cpp.o"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/han/utils/util.cpp.o -c /home/haihe/common/SubgraphMatching-master/utility/han/utils/util.cpp

utility/CMakeFiles/utility.dir/han/utils/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/han/utils/util.cpp.i"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haihe/common/SubgraphMatching-master/utility/han/utils/util.cpp > CMakeFiles/utility.dir/han/utils/util.cpp.i

utility/CMakeFiles/utility.dir/han/utils/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/han/utils/util.cpp.s"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haihe/common/SubgraphMatching-master/utility/han/utils/util.cpp -o CMakeFiles/utility.dir/han/utils/util.cpp.s

# Object files for target utility
utility_OBJECTS = \
"CMakeFiles/utility.dir/graphoperations.cpp.o" \
"CMakeFiles/utility.dir/commandparser.cpp.o" \
"CMakeFiles/utility.dir/computesetintersection.cpp.o" \
"CMakeFiles/utility.dir/bitsetoperation.cpp.o" \
"CMakeFiles/utility.dir/han/intersection_algos.cpp.o" \
"CMakeFiles/utility.dir/han/utils/util.cpp.o"

# External object files for target utility
utility_EXTERNAL_OBJECTS =

utility/libutility.so: utility/CMakeFiles/utility.dir/graphoperations.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/commandparser.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/computesetintersection.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/bitsetoperation.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/han/intersection_algos.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/han/utils/util.cpp.o
utility/libutility.so: utility/CMakeFiles/utility.dir/build.make
utility/libutility.so: utility/CMakeFiles/utility.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haihe/common/SubgraphMatching-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libutility.so"
	cd /home/haihe/common/SubgraphMatching-master/build/utility && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/utility.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utility/CMakeFiles/utility.dir/build: utility/libutility.so

.PHONY : utility/CMakeFiles/utility.dir/build

utility/CMakeFiles/utility.dir/clean:
	cd /home/haihe/common/SubgraphMatching-master/build/utility && $(CMAKE_COMMAND) -P CMakeFiles/utility.dir/cmake_clean.cmake
.PHONY : utility/CMakeFiles/utility.dir/clean

utility/CMakeFiles/utility.dir/depend:
	cd /home/haihe/common/SubgraphMatching-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haihe/common/SubgraphMatching-master /home/haihe/common/SubgraphMatching-master/utility /home/haihe/common/SubgraphMatching-master/build /home/haihe/common/SubgraphMatching-master/build/utility /home/haihe/common/SubgraphMatching-master/build/utility/CMakeFiles/utility.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utility/CMakeFiles/utility.dir/depend

