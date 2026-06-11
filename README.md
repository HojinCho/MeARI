# <span style="color:blue">Me</span><span style="color:mediumseagreen">A</span><span style="color:gold">R</span><span style="color:red">I</span>: <span style="color:blue">Me</span>sh-based <span style="color:mediumseagreen">A</span>GN <span style="color:gold">R</span>everberation <span style="color:red">I</span>ntegrator
<!-- # MeARI: Mesh-based AGN Reverberation Integrator -->
## Authors: Hojin Cho & Michael Fausnaugh
**Meari** (메아리 [me̞a̠ɾi]) means echo in Korean. Individual words consisting the acronym are not final and are subject to change to make more sense.

----

Uses 
 - Cython
 - OpenBLAS: 
    - `micromamba install openblas=*=*openmp*` if to use with openmp
    - `micromamba install openblas=*=*pthreads*` if to use with pthreads
    - Maybe need to install `libopenblas` seperately?
 - OpenMP (Optional)
 - CUDA (Optional)
 - [PCG](https://www.pcg-random.org/) (shipped along with the code)

Installation recommended using conda-forge instead of PyPI. https://pypackaging-native.github.io/key-issues/native-dependencies/blas_openmp/

### Requirements:
    - Cython
    - CBLAS (OpenBLAS in conda-forge is recommended.)
    - numpy (not required in backend, but for frontend.)