# <span style="color:blue">Me</span><span style="color:mediumseagreen">A</span><span style="color:gold">R</span><span style="color:red">I</span>: <span style="color:blue">Me</span>sh-based <span style="color:mediumseagreen">A</span>GN <span style="color:gold">R</span>everberation <span style="color:red">I</span>ntegrator
<!-- # MeARI: Mesh-based AGN Reverberation Integrator -->
## Authors: Hojin Cho & Michael Fausnaugh
**Meari** (메아리 [me̞a̠ɾi]) means echo in Korean. Individual words consisting the acronym are not final and are subject to change to make more sense.

----

This code models the ***reverberating surface***, such as the broad line region or the dusty torus, of AGNs to generate the transfer function of the responding light curves, i.e., broad emission lines or optical-infrared continua. Using and performs computes posterior function based on the light curves (either by sequential Bayesian or PRH-Q). The reverberating surface is modeled with triangular meshes as various geometry. Currently, the only adopted model is the thick disk model by [Goad et al. 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.426.3086G), which can also be used to model thin disks.
 
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