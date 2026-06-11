from setuptools import setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import os
import warnings


# from distutils.extension import Extension

package_name = 'meari'
src_path = 'meari/'

benchmark = True


# Following : https://stackoverflow.com/a/7071358/4755229
# Switch to : https://setuptools-scm.readthedocs.io/en/latest/ (https://stackoverflow.com/a/61960231/4755229 option 7)
import re
VERSIONFILE = src_path+"_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

use_cblas = True

use_cuda = False
use_fftw = False
# use_pocketfft = True
use_openmp = True
use_pthreads = False

ecompargs = ['-O3']
elinkargs = []
clear_symlinks = False


class OverrideWarning(Warning):
    pass

def conda_include():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return None, None
    conda_include = os.path.join(conda_prefix, 'include')
    conda_lib = os.path.join(conda_prefix, 'lib')
    return conda_include, conda_lib

def symlink_targeted_implementations(package_specs, target_dir=src_path+'extern/'):
    cwd = os.getcwd()
    os.chdir(target_dir)
    for k, package_spec in package_specs.items():
        dst = package_spec['dst']
        src = package_spec['src']
        try:
            os.symlink(src, dst)
        except FileExistsError:
            if os.path.islink(dst):
                os.unlink(dst)
                os.symlink(src, dst)
            else:
                warnings.warn(f"File {dst} exists in the target directory and will override configuration!", OverrideWarning)
    os.chdir(cwd)

def add_extra_spec(package_spec, extension_module: Extension):
    if package_spec['extra'] is not None:
        for k, v in package_spec['extra'].items():
            if not isinstance(v, list):
                v = [v]
            if k=='sources':
                extension_module.sources += v
            elif k=='include_dirs':
                extension_module.include_dirs += v
            elif k=='library_dirs':
                extension_module.library_dirs += v
            elif k=='libraries':
                extension_module.libraries += v


def unlink_targeted_implementations(package_specs, target_dir=src_path+'extern/'):
    cwd = os.getcwd()
    os.chdir(target_dir)
    for k, package_spec in package_specs.items():
        dst = package_spec['dst']
        try:
            os.unlink(dst)
        except:
            pass
    os.chdir(cwd)



if use_openmp:
    ecompargs_lia = ecompargs + ['-fopenmp']
    elinkargs_lia = elinkargs + ['-fopenmp']
elif use_pthreads:
    ecompargs_lia = ecompargs + ['-pthread']
    elinkargs_lia = elinkargs + ['-lpthread']
else:
    ecompargs_lia = ecompargs
    elinkargs_lia = elinkargs

if use_fftw:
    ecompargs_fft = ecompargs + ['-lfftw3']
    elinkargs_fft = elinkargs + ['-lfftw3']


import numpy
numpy_include = numpy.get_include()

# BLAS Module Selection
# class CustomBuildExt(build_ext):
#     user_options = build_ext.user_options + [
#         ('cblas=', None, 'Specify the name of BLAS library to use (available: openblas, blis).'),
#         ('cblas-include-dir=', None, 'Specify the include directory for the BLAS library.'),
#         ('cblas-lib-dir=', None, 'Specify the library directory for the BLAS library.'),
#     ]

#     def initialize_options(self):
#         super().initialize_options()
#         self.cblas = None
#         self.cblas_include_dir = None
#         self.cblas_lib_dir = None

#     def finalize_options(self):
#         super().finalize_options()

#         # Helper function to detect Conda BLAS
#         def detect_conda_blas(blastype):
#             conda_prefix = os.environ.get('CONDA_PREFIX')
#             if not conda_prefix:
#                 return None, None
#             conda_include = os.path.join(conda_prefix, 'include')
#             conda_lib = os.path.join(conda_prefix, 'lib')
#             if os.path.exists(conda_include) and os.path.exists(conda_lib):
#                 if blastype == 'openblas' and os.path.exists(os.path.join(conda_lib, 'libopenblas.so')):
#                     return conda_include, conda_lib
#                 if blastype == 'blis' and os.path.exists(os.path.join(conda_lib, 'libblis.so')):
#                     return conda_include, conda_lib
#             return None, None

#         # Logic to configure BLAS libraries
#         blas_include = None
#         blas_lib = None
#         blas_name = None

#         if self.cblas:
#             blas_name = self.cblas.lower()
#             if self.cblas_include_dir and self.cblas_lib_dir:
#                 blas_include = self.cblas_include_dir
#                 blas_lib = self.cblas_lib_dir
#             else:
#                 conda_include, conda_lib = detect_conda_blas(blas_name)
#                 if conda_include and conda_lib:
#                     blas_include, blas_lib = conda_include, conda_lib
#                 else:
#                     system_include = f"/usr/include/{blas_name}"
#                     system_lib = "/usr/lib"
#                     if os.path.exists(system_include) and os.path.exists(os.path.join(system_lib, f"lib{blas_name}.so")):
#                         blas_include, blas_lib = system_include, system_lib
#                     else:
#                         raise ValueError(f"{blas_name} not found in Conda or system paths. Please install it or provide paths explicitly.")
#         elif self.cblas_include_dir and self.cblas_lib_dir:
#             blas_include = self.cblas_include_dir
#             blas_lib = self.cblas_lib_dir
#         else:
#             conda_include, conda_lib = detect_conda_blas('openblas')
#             if conda_include and conda_lib:
#                 blas_name = 'openblas'
#                 blas_include, blas_lib = conda_include, conda_lib
#             else:
#                 raise ValueError("BLAS not found. Please install BLAS or provide its path explicitly.")

#         # Dynamically configure only BLAS-requiring extensions
#         for ext in self.extensions:
#             if ('REQUIRES_BLAS', 1) in ext.define_macros:
#                 print(f"Configuring BLAS for extension: {ext.name}")
#                 ext.include_dirs.append(blas_include)
#                 ext.library_dirs.append(blas_lib)
#                 ext.libraries.append(blas_name)

# FFT Module Selection
if False:
    fft_module = Extension(
        name=package_name + '.extern.fft',
        sources=[src_path + 'extern/pyfftw.pyx'],
        language_level=3,
        include_dirs=[
            # regarding pyfftw module.
        ],
        options={
            'build_ext': {
                'build_lib': src_path + 'extern',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    )
# else:
    
# Selecting Backend for LiA Module
extern_specs = { # Extern packages
    'lia':{'dst' : 'lia.pyx', 'src' : None, 'extra': None},
    'rng':{'dst' : 'rng.pyx', 'src' : None, 'extra': None},
    # 'fft':{'dst' : 'fft.pyx', 'src' : None, 'extra': None},
}
if use_cblas:
    extern_specs['lia']['src'] = 'lia/lia_cblas.pyx'
extern_specs['rng']['src'] = 'rng/rng.pyx'
# extern_specs['fft']['src'] = 'fft/fftw.pyx'

symlink_targeted_implementations(extern_specs, target_dir=src_path+'extern/')

lia_module = Extension(
    name=package_name + '.extern.lia',
    sources=[
        src_path + 'extern/lia.pyx'
    ],
    language_level=3,
    include_dirs = [src_path + 'extern/lia',],  # Will be dynamically populated
    # library_dirs=[],  # Will be dynamically populated
    libraries=['blas','lapack'],     # Will be dynamically populated
    options={
        'build_ext': {
            'build_lib': src_path + 'extern',
            'build_temp': 'build/temp',
        }
    },
    extra_compile_args = ecompargs_lia,
    extra_link_args    = elinkargs_lia,
    # define_macros=[('REQUIRES_BLAS', 1)],  # Metadata to mark BLAS requirement
)
# prhq_module = Extension(
#     package_name + '.stat.prhq',
#     sources=[src_path + 'stat/prhq_fft.pyx'],
#     language_level=3,
#     options={
#         'build_ext': {
#             'build_lib': src_path + 'stat',
#             'build_temp': 'build/temp',
#         }
#     },
#     extra_compile_args = ecompargs,
#     extra_link_args    = elinkargs,
# )
prhq_module = Extension(
    package_name + '.stat.prhq',
    sources=[src_path + 'stat/prhq_piecewise.pyx'],
    language_level=3,
    options={
        'build_ext': {
            'build_lib': src_path + 'stat',
            'build_temp': 'build/temp',
        }
    },
    extra_compile_args = ecompargs,
    extra_link_args    = elinkargs,
)
# fft_module = Extension(
#     name=package_name + '.extern.fft',
#     sources=[
#         src_path + 'extern/fft.pyx', 
#     ],
#     language_level=3,
#     include_dirs=[
#         # numpy_include, 
#         # src_path + 'extern/fft/pocketfft',
#     ],
#     libraries=['fftw3'],
#     options={
#         'build_ext': {
#             'build_lib': src_path + 'extern',
#             'build_temp': 'build/temp',
#         }
#     },
#     extra_compile_args = ecompargs_fft,
#     extra_link_args    = elinkargs_fft,
# )
# # add_extra_spec(extern_specs['fft'], fft_module)

extensions = [
##### Utils
    Extension(
        package_name + '.utils.numpy_interface',
        sources=[src_path + 'utils/numpy_interface.pyx'],
        language_level=3,
        include_dirs=[numpy_include],
        options={
            'build_ext': {
                'build_lib': src_path + 'utils',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.utils.algorithms',
        sources=[src_path + 'utils/algorithms.pyx'],
        language_level=3,
        include_dirs=[numpy_include],
        options={
            'build_ext': {
                'build_lib': src_path + 'utils',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.utils.interpolation',
        sources=[src_path + 'utils/interpolation.pyx'],
        language_level=3,
        # include_dirs=[numpy_include],
        options={
            'build_ext': {
                'build_lib': src_path + 'utils',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
##### External Dependency Codes
    # fft_module,
    lia_module,
    Extension(
        package_name + '.extern.rng',
        sources=[src_path + 'extern/rng.pyx'],
        language_level=3,
        include_dirs=[
            src_path + 'extern/rng',
            src_path + 'extern/rng/pcg',
        ],
        options={
            'build_ext': {
                'build_lib': src_path + 'extern',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
##### Mesh
    Extension(
        package_name + '.mesh.transformables',
        sources=[src_path + 'mesh/transformables.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'mesh',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.mesh.diskmesh_auxil',
        sources=[src_path + 'mesh/diskmesh_auxil.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'mesh',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.mesh.compute_tf',
        sources=[src_path + 'mesh/compute_tf.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'mesh',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.mesh.diskmesh',
        sources=[src_path + 'mesh/diskmesh.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'mesh',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
##### Geometry
    Extension(
        package_name + '.geometry.diskgeom',
        sources=[src_path + 'geometry/diskgeom.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'geometry',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.geometry.slim_free',
        sources=[src_path + 'geometry/slim_free.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'geometry',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.geometry.thin_goad',
        sources=[src_path + 'geometry/thin_goad.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'geometry',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
##### Physics
    Extension(
        package_name + '.physics.kinematics',
        sources=[src_path + 'physics/kinematics.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'physics',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.physics.response',
        sources=[src_path + 'physics/response.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'physics',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.physics.thermal',
        sources=[src_path + 'physics/thermal.pyx'],
        language_level=3,
        include_dirs=[numpy_include],
        options={
            'build_ext': {
                'build_lib': src_path + 'physics',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
##### Statistics
    Extension(
        package_name + '.stat.lightcurves',
        sources=[src_path + 'stat/lightcurves.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'stat',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.stat.variability',
        sources=[src_path + 'stat/variability.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'stat',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    prhq_module,
    Extension(
        package_name + '.stat.stats',
        sources=[src_path + 'stat/stats.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'stat',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
    Extension(
        package_name + '.stat.likelihoods',
        sources=[src_path + 'stat/likelihoods.pyx'],
        language_level=3,
        options={
            'build_ext': {
                'build_lib': src_path + 'stat',
                'build_temp': 'build/temp',
            }
        },
        extra_compile_args = ecompargs,
        extra_link_args    = elinkargs,
    ),
]
if benchmark:
    extensions += [
        Extension(
            package_name + '.utils.benchmark',
            sources=[src_path + 'utils/benchmark.pyx'],
            language_level=3,
            # include_dirs=[numpy_include],
            options={
                'build_ext': {
                    'build_lib': src_path + 'utils',
                    'build_temp': 'build/temp',
                }
            },
            extra_compile_args = ecompargs,
            extra_link_args    = elinkargs,
        ),
    ]

setup(
    name=package_name,
    version=verstr, # https://peps.python.org/pep-0440/#pre-releases
    include_package_data=True, # https://sixty-north.com/blog/including-package-data-in-python-packages.html
    # package_dir={"": package_name},
    ext_modules=cythonize(
        extensions, 
        gdb_debug=False, 
        # gdb_debug=True, emit_linenums=True,
        annotate=False,
    ),
    packages=[
        package_name, 
        package_name + '.utils', 
        package_name + '.extern',
        package_name + '.physics', 
        package_name + '.geometry', 
        package_name + '.stat', 
        package_name + '.mesh',
    ],
    zip_safe=False,
)
if clear_symlinks:
    unlink_targeted_implementations(extern_specs, target_dir=src_path+'extern/')