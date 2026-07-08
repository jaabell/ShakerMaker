# building environment for stampede3 should be done in a idev mode
# idev 

# after the idev command, you will be in a new shell
clear
ml phdf5
ml python/3.9.18


# build an envirormrnt and activate it (not necessary, but recommended)

ENVNAME=ShakerMaker_env
# ENVNAME=ShakerMakerTestingEnv

python3 -m venv $ENVNAME
source $ENVNAME/bin/activate


# installing the dependencies
pip install numpy==1.23.0 scipy==1.11.1 matplotlib geopy pyproj tqdm h5py mpi4py



# install the ShakerMaker package
# clone the repository
git clone https://github.com/amnp95/ShakerMaker.git
cd ShakerMaker
cp tacc/stampede3.py .

# set the environment variables for the Intel compiler
export CC=icx
export CXX=icpx
export FC=ifx
export F77=ifx
export F90=ifx

# build the package
python3 stampede3.py build
python3 stampede3.py install



# pip list : you should see the ShakerMaker package in the list of installed packages
pip list
cd ..


