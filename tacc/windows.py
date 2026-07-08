import os
from numpy.distutils.core import Extension, setup

# Project information
name = "shakermaker"
version = "0.1"
release = "0.01"
author = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren"

# Source files
srcdir = "shakermaker/core/"
srcs = [
    srcdir + 'core.pyf',
    srcdir + "subgreen.f",
    srcdir + "subgreen2.f",
    srcdir + "subfocal.f",
    srcdir + "subfk.f",
    srcdir + "subtrav.f",
    srcdir + "tau_p.f",
    srcdir + "kernel.f",
    srcdir + "prop.f",
    srcdir + "source.f",
    srcdir + "bessel.f",
    srcdir + "haskell.f",
    srcdir + "fft.c",
    srcdir + "Complex.c"
]

# Set Intel compilers for C and Fortran
os.environ['FC'] = 'ifx'  # Intel Fortran Compiler
os.environ['CC'] = 'icx'  # Intel C Compiler
os.environ['CXX'] = 'icx'  # Intel C++ Compiler
os.environ['F77'] = 'ifx'  # Set Fortran 77 compiler



# Compiler flags for Intel compilers
extra_f77_compile_args = [
    "/extend-source:132",   # For Fortran line length compatibility
    "/Qparallel",           # Parallel compilation
    "/O1",                  # Optimization level 3
    "/Qopenmp",             # OpenMP support
]

extra_compile_args = [
    "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "/QxHost",              # Optimize for host machine's processor
    "/O1",                  # Aggressive optimization
    "/Qopenmp",             # Enable OpenMP for parallel processing
]

# Define extension modules
ext_modules = [
    Extension(
        name='shakermaker.core',
        sources=srcs,
        extra_f77_compile_args=extra_f77_compile_args,
        extra_compile_args=extra_compile_args,
        include_dirs=["D:/Programs/Intel/oneAPI/include"],  # Adjust to your include path
        library_dirs=["D:/Programs/Intel/oneAPI/lib"]       # Adjust to your library path
    )
]

# Define setup configuration
setup(
    name=name,
    package_dir={'shakermaker': 'shakermaker'},
    packages=[
        "shakermaker",
        "shakermaker.cm_library",
        "shakermaker.sl_extensions",
        "shakermaker.slw_extensions",
        "shakermaker.stf_extensions",
        "shakermaker.tools",
    ],
    ext_modules=ext_modules,
    version=version,
    description="README.md",
    author=author,
    author_email="info@joseabell.com",
    url="http://www.joseabell.com",
    download_url="tbd",
    keywords=["earthquake", "engineering", "drm", "simulation"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Fortran",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    long_description="""
    shakermaker
    -------------------------------------
    Create realistic seismograms!
    """,
)
