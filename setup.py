# NOTE: the Fortran/C extensions are built with f2py via numpy.distutils,
# which was removed from Python's packaging tooling in Python 3.12. This
# setup.py therefore only works on Python 3.8-3.11; building on Python >= 3.12
# requires migrating to meson-python / scikit-build.
from setuptools import setup
from numpy.distutils.core import Extension, setup as np_setup
import importlib.util
import os
import sys
import shutil
import subprocess

from setuptools.command.install import install as _install

on_rtd = os.environ.get('READTHEDOCS') == 'True'

name    = "shakermaker"

# Single source of truth for the package version: shakermaker/version.py
_version_globals = {}
with open(os.path.join(os.path.dirname(__file__), "shakermaker", "version.py")) as _vf:
    exec(_vf.read(), _version_globals)
version = _version_globals["shakermaker_version"]

release = "0.01"
author  = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren"

profile = False

srcdir = "shakermaker/core/"
ffiles = [
    "subgreen.f", "subgreen2.f", "subfocal.f", "subfk.f", "subtrav.f",
    "tau_p.f", "kernel.f", "prop.f", "source.f", "bessel.f", "haskell.f",
    "fft.c", "Complex.c",
]
srcs = [srcdir + "core.pyf"] + [srcdir + f for f in ffiles]

if on_rtd:
    ext_modules = []
else:
    if profile:
        ext1 = Extension(
            name='shakermaker.core',
            sources=srcs,
            extra_f77_compile_args=[
                "-ffixed-line-length-132", "-Wno-tabs",
                "-Wno-unused-dummy-argument", "-pg", "-fopenmp",
            ],
            extra_link_args=["-pg", "-fopenmp"],
        )
    else:
        # Windows change: Intel ifx compiler flags (via intelvem f2py backend)
        if sys.platform == 'win32':
            f77_args     = ["/Qopenmp", "/extend-source:132"]
            link_args    = ["/Qopenmp", "/F67108864"]
            compile_args = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
        else:
            f77_args     = [
                "-ffixed-line-length-132", "-Wno-tabs",
                "-Wno-unused-dummy-argument", "-fPIC", "-fopenmp",
            ]
            link_args    = ["-fopenmp"]
            compile_args = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

        ext1 = Extension(
            name='shakermaker.core',
            sources=srcs,
            extra_f77_compile_args=f77_args,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )

    ext_modules = [ext1]


# =============================================================================
# FFSP compilation support
# =============================================================================

def compile_ffsp():
    """Compile the FFSP Fortran wrapper using f2py."""
    ffsp_dir = os.path.join(os.path.dirname(__file__), 'shakermaker', 'ffsp')
    if sys.platform == 'win32':
        _compile_ffsp_windows(ffsp_dir)
    else:
        _compile_ffsp_linux(ffsp_dir)


def _compile_ffsp_linux(ffsp_dir):
    """Compile FFSP on Linux / macOS using gfortran via f2py."""
    fortran_sources = [
        "ffsp_wrapper.f90", "ffsp_comm.f90", "spfield_n.f90",
        "dcf_subs_1.f90", "slip_rate.f90", "ffsp_tool.f",
    ]
    for f in os.listdir(ffsp_dir):
        if f.startswith('ffsp_core') and (f.endswith('.so') or f.endswith('.pyd')):
            os.remove(os.path.join(ffsp_dir, f))

    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "ffsp.pyf",
        *fortran_sources,
        "--f90flags=-O3 -fPIC",
        "--f77flags=-O3 -std=legacy -fPIC",
        "-m", "ffsp_core",
    ]
    print("[ffsp] compiling on Linux/macOS...")
    result = subprocess.run(cmd, cwd=ffsp_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("[ffsp] compilation failed -- see output above")

    so_files = [f for f in os.listdir(ffsp_dir)
                if f.startswith('ffsp_core') and f.endswith('.so')]
    if so_files:
        print(f"[OK] FFSP compiled (Linux): {so_files[0]}")
    else:
        raise RuntimeError("FFSP .so not found after compilation")


def _compile_ffsp_windows(ffsp_dir):
    """Compile FFSP on Windows using Intel ifx via f2py intelvem backend.

    capture_output=False so ifx compiler errors print directly to console.
    """
    fortran_sources = [
        "ffsp_wrapper.f90", "ffsp_comm.f90", "spfield_n.f90",
        "dcf_subs_1.f90", "slip_rate.f90", "ffsp_tool.f",
    ]
    for f in os.listdir(ffsp_dir):
        if f.startswith('ffsp_core') and (f.endswith('.pyd') or f.endswith('.so')):
            os.remove(os.path.join(ffsp_dir, f))

    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "ffsp.pyf",
        *fortran_sources,
        "--fcompiler=intelvem",
        "--f90flags=/Qopenmp",
        "--f77flags=/Qopenmp",
        "-m", "ffsp_core",
    ]
    result = subprocess.run(cmd, cwd=ffsp_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[ffsp] f2py/ifx failed (exit {result.returncode}) "
            "-- read the ifx error lines above this message.")

    pyd_files = [f for f in os.listdir(ffsp_dir)
                 if f.startswith('ffsp_core') and f.endswith('.pyd')]
    if pyd_files:
        print(f"[OK] FFSP compiled (Windows): {pyd_files[0]}")
    else:
        raise RuntimeError(
            "FFSP .pyd not found after compilation despite returncode=0")


def _install_ffsp_binary(ffsp_dir):
    """Copy compiled .so/.pyd into site-packages after install."""
    import site
    ext = ".pyd" if sys.platform == "win32" else ".so"
    for sp in site.getsitepackages():
        dest = os.path.join(sp, "shakermaker", "ffsp")
        if os.path.isdir(dest):
            for f in os.listdir(ffsp_dir):
                if f.startswith("ffsp_core") and f.endswith(ext):
                    shutil.copy2(os.path.join(ffsp_dir, f),
                                 os.path.join(dest, f))
                    print(f"[ffsp] installed {f} -> {dest}")
            return


class PostInstallCommand(_install):
    """Custom install command: compile FFSP after the main install."""

    def run(self):
        _install.run(self)
        ffsp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shakermaker", "ffsp")
        compile_ffsp()
        _install_ffsp_binary(ffsp_dir)


# =============================================================================

found_sphinx    = importlib.util.find_spec('sphinx') is not None
cmdclass        = {'install': PostInstallCommand}
command_options = {}

np_setup(
    name    = name,
    version = version,
    author  = author,
    author_email  = "info@joseabell.com",
    url           = "http://www.joseabell.com",
    description   = "Physics-based earthquake ground motion synthesis (FK Green's functions + DRM).",
    long_description = """\
        shakermaker
        -------------------------------------

        Create realistic seismograms!

        """,
    keywords = ["earthquake", "engineering", "drm", "simulation"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Fortran",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: GPL",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires = ">=3.8,<3.12",
    package_dir = {'shakermaker': 'shakermaker'},
    packages = [
        "shakermaker",
        "shakermaker.cm_library",
        "shakermaker.sl_extensions",
        "shakermaker.slw_extensions",
        "shakermaker.stf_extensions",
        "shakermaker.tools",
        "shakermaker.ffsp",
        "shakermaker.sw4_exporter",
        "shakermaker.crust1",
    ],
    package_data = {
        'shakermaker.ffsp': [
            'ffsp_core.cpython-*.so',
            'ffsp_core.cpython-*.pyd',
            '*.f90', '*.f',
            'makefile', 'Makefile_f2py',
            'ffsp.pyf',
        ],
        'shakermaker.crust1': ['crust1.0/*'],
    },
    ext_modules     = ext_modules,
    cmdclass        = cmdclass,
    command_options = command_options,
)
