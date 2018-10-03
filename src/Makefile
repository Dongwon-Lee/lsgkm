CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
#CFLAGS = -Wall -Wconversion -g -fPIC
SHVER = 2

all: gkmtrain gkmpredict gkmexplain
#all: gkmtrain gkmpredict gkmmatrix 

gkmexplain: gkmexplain.c libsvm.o libsvm_gkm.o
	$(CXX) $(CFLAGS) gkmexplain.c libsvm.o libsvm_gkm.o -o gkmexplain -lm -lpthread
gkmpredict: gkmpredict.c libsvm.o libsvm_gkm.o
	$(CXX) $(CFLAGS) gkmpredict.c libsvm.o libsvm_gkm.o -o gkmpredict -lm -lpthread
gkmtrain: gkmtrain.c libsvm.o libsvm_gkm.o
	$(CXX) $(CFLAGS) gkmtrain.c libsvm.o libsvm_gkm.o -o gkmtrain -lm -lpthread

#for debug
#gkmmatrix: gkmmatrix.c libsvm.o libsvm_gkm.o
#	$(CXX) $(CFLAGS) gkmmatrix.c libsvm.o libsvm_gkm.o -o gkmmatrix -lm -lpthread

libsvm.o: libsvm.cpp libsvm.h
	$(CXX) $(CFLAGS) -c libsvm.cpp
libsvm_gkm.o: libsvm_gkm.c libsvm_gkm.h
	$(CXX) $(CFLAGS) -c libsvm_gkm.c

clean:
	rm -f libsvm.o libsvm_gkm.o gkmtrain gkmpredict gkmexplain gkmmatrix

install:
	cp gkmtrain gkmpredict gkmexplain ../bin
