CXXFLAGS := -g -std=c++17 -Wall -lpthread -O3 
CXX := g++

# ifeq (/usr/bin/clang++-11,$(wildcard /usr/bin/clang++-11*))
#     CXX=clang++-11
# endif

all: pi.out

pi.out: pi.cpp
	$(CXX) $(CXXFLAGS) -o pi.out pi.cpp 

.PHONY:  clean
clean:
	-rm -f pi.out