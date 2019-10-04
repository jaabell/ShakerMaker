FFLAGS=-O1
CFLAGS=-O1
FC=gfortran -ffixed-line-length-132 -fPIC
CC=gcc -fPIC
PYF=syn.pyf 
F2PY=f2py -c ${PYF} 

OBJS = subfocal.o subfk.o subtrav.o tau_p.o kernel.o prop.o source.o bessel.o haskell.o fft.o Complex.o 

all: fksubs synmodule

synmodule: fksubs
	${F2PY} ${OBJS}

fksubs:  ${OBJS} 

clean:
	rm -f *.o *.so
