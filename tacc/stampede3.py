from numpy.distutils.core import Extension, setup


name = "shakermaker"
version = "0.1"
release = "0.01"
author = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren"

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

# Minimal flags for compatibility
ext_modules = [
    Extension(
        name='shakermaker.core',
        sources=srcs,
        extra_f77_compile_args=[
            "-132",            # Fixed format 132 character line length
            "-w",              # Disable warnings
            "-fPIC",           # Position Independent Code
            "-O2",             # Moderate optimization level
        ],
        extra_compile_args=[
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            "-O2",             # Moderate optimization level
            "-fPIC"            # Position Independent Code
        ]
    )
]

cmdclass = {}
command_options = {}

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
        "Development Status :: Released",
        "Environment :: Other Environment",
        "Intended Audience :: Savvy Earthquake Engineers",
        "License :: GPL",
        "Operating System :: OS Independent",
        "Topic :: TBD",
        "Topic :: TBD2",
    ],
    long_description="""\
                            shakermaker
                            -------------------------------------
                            Create realistic seismograms!

                            """,
    cmdclass=cmdclass,
    command_options=command_options,
)
