#FILE(GLOB_RECURSE CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

FIND_PACKAGE(HDF5 COMPONENTS HL REQUIRED)

INCLUDE_DIRECTORIES(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})

#ADD_EXECUTABLE(fixsy FixedSy.cpp
#                    FixedW.cpp
#                    Main.cpp
#                    Word2Vec.cpp
#                    CommonFunction.cpp
#                    ValueDef1.cpp)
ADD_EXECUTABLE(fixw solver.c 
                    data.c
                    hierarchical_softmax.c)

#TARGET_LINK_LIBRARIES(fixsy pthread)
TARGET_LINK_LIBRARIES(fixw pthread m ${HDF5_LIBRARIES})
