python -m venv myenv
call .\myenv\Scripts\activate
call pip install setuptools
call pip install numpy==1.23.5
call pip install scipy==1.11.4
call pip install matplotlib
call pip install geopy
call pip install pyproj
call pip install tqdm
call pip install h5py
call pip install mpi4py

@REM replace the path with your own path to the Intel oneAPI installation
call "D:\Programs\Intel\oneAPI\setvars.bat"
python IntelWindowsSetup.py build
python IntelWindowsSetup.py install

