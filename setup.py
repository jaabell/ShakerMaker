# from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import imp 

name = "shakermaker"
version = "0.1"
release = "0.01"
author = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren",

profile = False

srcdir = "shakermaker/core/"
ffiles = ["subgreen.f", "subfocal.f","subfk.f", "subtrav.f", "tau_p.f", "kernel.f", "prop.f", "source.f", "bessel.f", "haskell.f", "fft.c", "Complex.c"]
srcs = [srcdir+'core.pyf']
for f in ffiles:
    srcs.append(srcdir+f)


if profile:
    ext1 = Extension(name = 'shakermaker.core',
                     sources = srcs, 
                     extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-pg"],
                     extra_link_args=["-pg"])
else:
    ext1 = Extension(name = 'shakermaker.core',
                         sources = srcs, 
                         extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument","-fPIC"],
                         extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"])


try:
    imp.find_module('sphinx')
    found_sphinx = True
except ImportError:
    found_sphinx = False

cmdclass = {}
command_options = {}

if found_sphinx:
    print("Configuring Sphinx autodocumentation")
    from sphinx.setup_command import BuildDoc
    
    cmdclass['build_sphinx'] =  BuildDoc
    command_options['build_sphinx'] =  {
                'project': ('setup.py', name),
                'version': ('setup.py', version),
                'release': ('setup.py', release),
                'source_dir': ('setup.py', 'docs')}



setup(
    name = name,
    package_dir = {
        'shakermaker' : 'shakermaker'
    },
    packages = [
        "shakermaker",
        "shakermaker.sl_extensions",
        "shakermaker.stf_extensions",
        # "shakermaker.Sources",
        # "shakermaker.SourceTimeFunctions",
        # "shakermaker.Tools"
        ],
    ext_modules = [ext1],
    version = version,
    description = "README.md",
    author = author,
    author_email = "info@joseabell.com",
    url = "http://www.joseabell.com",
    download_url = "tbd",
    keywords = ["earthquake", "engineering", "drm", "simulation"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Fortran",
        "Development Status :: Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Savvy Eartkquake Engineers",
        "License :: GPL",
        "Operating System :: OS Independent",
        "Topic :: TBD",
        "Topic :: TBD2",
        ],
    long_description = """\
        shakermaker
        -------------------------------------
        
        Simulate the world!
        
        """,
    cmdclass=cmdclass,
    command_options=command_options,
)

