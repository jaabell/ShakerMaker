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


# ext_modules  = [
#     Extension (
#         name='shakermaker.core',
#         sources=srcs,
#         extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-fPIC"],
#         extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
#     )
# ]


ext_modules  = [
    Extension (
        name='shakermaker.core',
        sources=srcs,
        extra_f77_compile_args=["-132",
                                "-w",
                                "-fPIC",
                                "-O3",
                                "-xHost",
                                "-qopenmp",
                                "-ipo"
                                ],
        extra_compile_args=[
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            "-xHost",          # Optimize for the host machine's processor
            "-O3",             # Aggressive optimizations
            "-qopenmp"         # If you are using OpenMP, enables parallelization
             ]
    )
 ]



'''

ext_modules = [
    Extension(
        name='shakermaker.core',
        sources=srcs,
        extra_f77_compile_args=[
            "-O1",             # Safe optimization level: balances performance and stability
            "-xHost",          # Optimizes for the current machine's processor
            "-qopenmp",        # Enables parallelization using OpenMP (if the code supports it)
            "-ipo",            # Interprocedural optimization across multiple files for better performance
            "-fPIC"            # Ensures position-independent code for shared libraries
        ],
        extra_compile_args=[
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",  # Ensures compatibility with older NumPy APIs
            "-xHost",          # Processor-specific optimizations for maximum performance
            "-O1",             # Safe optimization level
            "-qopenmp"         # Enables parallelization
        ]
    )
]

'''







# ext_modules  = [
#     Extension (
#         name='shakermaker.core',
#         sources=srcs,
#         extra_f77_compile_args=[
#             "-extend-source",  # Equivalent to -ffixed-line-length-132
#             "-w",              # Suppresses all warnings, including tabs and unused dummy arguments
#             "-fPIC"            # Position-independent code, same as with GCC
#         ],
#         extra_compile_args=[
#             "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
#             # "-xHost",          # Optimize for the host machine's processor
#             # "-O3",             # Aggressive optimizations
#             # "-qopenmp"         # If you are using OpenMP, enables parallelization
#         ]
#     )
# ]




cmdclass = {}
command_options = {}

setup(
    name             =  name,
    package_dir      = {'shakermaker': 'shakermaker'},
    packages         = [
                            "shakermaker",
                            "shakermaker.cm_library",
                            "shakermaker.sl_extensions",
                            "shakermaker.slw_extensions",
                            "shakermaker.stf_extensions",
                            "shakermaker.tools",
                        ],
    ext_modules      = ext_modules,
    version          = version,
    description      = "README.md",
    author           = author,
    author_email     = "info@joseabell.com",
    url              = "http://www.joseabell.com",
    download_url     = "tbd",
    keywords         = ["earthquake", "engineering", "drm", "simulation"],
    classifiers      = [
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
    long_description =  """\
                            shakermaker
                            -------------------------------------
                            Create realistic seismograms!

                            """,
    cmdclass         = cmdclass,
    command_options  = command_options,
)
