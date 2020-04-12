#
# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.10)

project(Video_Processing_Framework)

set(TRACK_TOKEN_ALLOCATIONS FALSE CACHE BOOL "Debug memory allocations within VPF")

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

if(UNIX)
	set (CMAKE_CXX_STANDARD 11)
	include_directories(/usr/local/cuda/include)
	link_directories(/usr/local/cuda/lib64)
endif(UNIX)

add_subdirectory(PyNvCodec)
add_subdirectory(PytorchNvCodec)

#Add C++ sample
include_directories(${TC_CORE_INC_PATH})
include_directories(${TC_INC_PATH})
include_directories(${VIDEO_CODEC_SDK_INCLUDE_DIR})

#Bring all the stuff together
install(FILES $<TARGET_FILE:TC_CORE>									DESTINATION bin)
install(FILES $<TARGET_FILE:TC>											DESTINATION bin)

if(GENERATE_PYTHON_BINDINGS)
	install(FILES $<TARGET_FILE:PyNvCodec>								DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleRTSP.py				DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDecode.py			DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleEncode.py			DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTranscode.py		DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleFrameUpload.py		DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleColorConversion.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleSufraceDownload.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTranscodeOneToN.py	DESTINATION bin)
endif(GENERATE_PYTHON_BINDINGS)

if(GENERATE_PYTORCH_EXTENSION)
	#Extension will be built using torch.utils.cpp_extension;
	#So we just launch python script;
	add_custom_command(	OUTPUT Pytorch_Nv_Codec
						COMMAND cd ${PYTORCH_EXTENSION_SOURCES_DIR} && python setup.py build --build-lib="${CMAKE_INSTALL_PREFIX}/bin")

	add_custom_target(	PytorchNvCodec
						DEPENDS Pytorch_Nv_Codec)
	add_dependencies(PyNvCodec PytorchNvCodec)
endif(GENERATE_PYTORCH_EXTENSION)

set(SRC_FILE_EXTENSIONS c cpp)

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
	file(GLOB FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
	set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
endforeach()

set(SAMPLE_SOURCES ${LINT_SOURCES})

foreach(SAMPLE_NAME ${SAMPLE_SOURCES})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SAMPLE_NAME})
	list(GET SEXY_LIST -1 TEST_CASE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
	list(GET SEXY_LIST 0 TEST_CASE_NAME)
	add_executable(${TEST_CASE_NAME}  ${SAMPLE_NAME})
	target_link_libraries(${TEST_CASE_NAME} TC opencv_core opencv_highgui)
	set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
endforeach()
