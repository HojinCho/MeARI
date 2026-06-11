# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, realloc, calloc, free
from libc.string cimport strlen, strcmp, strcpy, memcpy

from ..utils.numpy_interface cimport (
    numpy_dbl_1d,
    numpy_dbl_2d,
)

from ..extern.lia cimport ( # BLAS and LAPACK
    # Unary
    lia_v_norm, lia_v_norm_A_i, lia_v_comp, 
    lia3d_v_rho_R,
    # Binary
    lia_sv_scalar_inplace, lia_sv_scalar_A_ii_inplace, lia_sv_scalar, lia_sv_scalar_A_ii,
    lia_vv_dot, lia_vv_dot_A_0i, lia_vv_dot_A_ii, lia_vv_cpy, lia_vv_cpy_A_ii, lia_vv_swp, lia_vv_swp_A_ii,
    lia_vv_add_inplace, lia_vv_add, lia_vv_add_A_i0_inplace, lia_vv_add_A_i0, lia_vv_add_A_ii_inplace, lia_vv_add_A_ii,
    lia_vv_sub_inplace, lia_vv_sub, lia_vv_sub_A_i0_inplace, lia_vv_sub_A_i0, lia_vv_sub_A_ii_inplace, lia_vv_sub_A_ii,
    lia_vv_mul_inplace, lia_vv_mul, lia_vv_mul_A_i0_inplace, lia_vv_mul_A_i0, lia_vv_mul_A_ii_inplace, lia_vv_mul_A_ii,
    lia_vv_div_inplace, lia_vv_div, lia_vv_div_A_i0_inplace, lia_vv_div_A_i0, lia_vv_div_A_ii_inplace, lia_vv_div_A_ii,
    lia_mv_mul, lia_mv_mul_A_0i, lia_mv_mul_A_ii,
    lia2d_mv_rot_inplace, lia2d_mv_rot_A_0i_inplace, lia2d_mv_rot, lia2d_mv_rot_A_0i, lia3d_mv_rot_inplace, lia3d_mv_rot_A_0i_inplace, lia3d_mv_rot, lia3d_mv_rot_A_0i,
    # Ternary
    lia_vmv_quadform_bf, lia_vmv_quadform_A_i0i_bf, lia_vmv_quadform_A_iii_bf, 
    lia_vmv_quadform, lia_vmv_quadform_A_i0i, lia_vmv_quadform_A_iii, 
    # Quarternary 
    lia_svsv_lincomb, lia_svsv_lincomb_A_0i0i, lia_svsv_lincomb_A_iiii,
)



###
# General-purpose n-dimensional physical vector space based on Cython.
# In theory, this can be used even for pseudo-metric spaces, such as Minkowski metric.
# May add dyads and higher rank tensors if necessary.
# May need to implement BLAS and LAPACK for matrix operations?
# https://pub.curvenote.com/019072c5-0119-76fe-b505-1577531ab450/public/ian_henriksen-61bd76e27d8d27566a7e02ec890094e5.pdf
###

cdef void rotate_vector(long ndata, long ndim, double *rotmat, double * vec_in, double * vec_out) noexcept nogil:
    lia_mv_mul_A_0i(vec_out, rotmat, vec_in, ndim, ndim, ndata, T=False) 

cdef void rotate_vector_T(long ndata, long ndim, double *rotmat, double * vec_in, double * vec_out) noexcept nogil:
    lia_mv_mul_A_0i(vec_out, rotmat, vec_in, ndim, ndim, ndata, T=True) 

cdef void inner_prod_const(long ndata, long ndim, double * vec, double * vec_const, double * out) noexcept nogil:
    lia_vv_dot_A_0i(out, vec_const, vec, ndim, ndata)

cdef void inner_prod(long ndata, long ndim, double * vec1, double * vec2, double * out) noexcept nogil:
    lia_vv_dot_A_ii(out, vec1, vec2, ndim, ndata)

cdef void inner_prod_xyz(long ndata, long ndim, double * vec, double * xyz, double * R, double * out) noexcept nogil:
    lia_vv_dot_A_ii(out, vec, xyz, ndim, ndata)
    lia_vv_div_inplace(out, R, ndata)

cdef void generalized_inner_prod(
    long ndata, long ndim, double * metric, double * vec1, double * vec2, double * out
) noexcept nogil:
    lia_vmv_quadform_A_i0i(out, vec1, metric, vec2, ndim, ndim, ndata)
    # cdef long i, j, J, k, K
    # for k in prange(ndata, nogil=True): # Use BLAS
    #     out[k] = 0
    #     K = ndim*k
    #     for j in range(ndim):
    #         J = ndim*j
    #         for i in range(ndim):
    #             out[k] += vec1[K + j]*metric[J + i]*vec2[K + i]

cdef class StrLists():
    def __cinit__(self, ):
        self.__current_buffer_len = 3
        self.N = 0
        self.len = <long *>malloc(self.__current_buffer_len*sizeof(long))
        self.data = <char **>malloc(self.__current_buffer_len*sizeof(char *))
    
    def __dealloc__(self):
        cdef long i
        if self.N > 0:
            for i in range(self.N):
                if self.data[i] != NULL:
                    free(self.data[i])
        if self.data != NULL:
            free(self.data)
        if self.len != NULL:
            free(self.len)

    cpdef void add(self, str s_str) noexcept:
        b = s_str.encode('utf-8')
        cdef char *s = b
        cdef long i
        with nogil:
            if self.N == self.__current_buffer_len:
                self.__current_buffer_len *= 2
                self.len = <long *>realloc(self.len, self.__current_buffer_len*sizeof(long))
                self.data = <char **>realloc(self.data, self.__current_buffer_len*sizeof(char *))
            self.len[self.N] = strlen(s)
            self.data[self.N] = <char *>malloc((self.len[self.N]+1)*sizeof(char))
            strcpy(self.data[self.N], s)
            self.N += 1

    cpdef void remove(self, str s_str) noexcept:
        b = s_str.encode('utf-8')
        cdef char *s = b
        cdef long i
        with nogil:
            for i in range(self.N):
                if self.len[i] == strlen(s) and strcmp(self.data[i], s) == 0:
                    free(self.data[i])
                    for i in range(i, self.N-1):
                        self.len[i] = self.len[i+1]
                        self.data[i] = self.data[i+1]
                    self.N -= 1
                    break

    cpdef bint is_in(self, str s_str) noexcept:
        b = s_str.encode('utf-8')
        cdef char *s = b
        cdef long i
        with nogil:
            for i in range(self.N):
                if self.len[i] == strlen(s) and strcmp(self.data[i], s) == 0:
                    return True
            return False
    
    cpdef long get_index(self, str s_str) noexcept:
        b = s_str.encode('utf-8')
        cdef char *s = b
        cdef long i
        with nogil:
            for i in range(self.N):
                if self.len[i] == strlen(s) and strcmp(self.data[i], s) == 0:
                    return i
            return -1

    cpdef str get(self, long i):
        if i < 0 or i >= self.N:
            raise IndexError("Index out of bounds")
        return self.data[i][:self.len[i]].decode('utf-8', 'strict')

    cpdef list to_list(self):
        cdef long i
        return [self.data[i][:self.len[i]].decode('utf-8', 'strict') for i in range(self.N)]

@cython.final
cdef class Scalars():
    def __cinit__(self, long ndata, long N=4):
        cdef long i
        self.__current_buffer_len = N
        self.N = 0
        self.ndata = ndata
        self.size = ndata*sizeof(double)
        self.entries = StrLists.__new__(StrLists)
        self.__data = <double **>malloc(self.__current_buffer_len*sizeof(double *))
        
    def __dealloc__(self):
        cdef long i
        if self.N > 0:
            for i in range(self.N):
                if self.__data[i] != NULL:
                    free(self.__data[i])
        if self.__data != NULL:
            free(self.__data)

    cpdef object get_by_index(self, long i):
        return numpy_dbl_1d(self.__data[i], self.ndata, False)

    cpdef object get(self, str s_str):
        cdef long i = self.entries.get_index(s_str)
        if i == -1:
            raise ValueError("No such entry")
        return numpy_dbl_1d(self.__data[i], self.ndata, False)

    cpdef void allocate_entry(self, str s_str) noexcept nogil:
        cdef bint in_entry
        with gil:
            in_entry = self.entries.is_in(s_str)
        if not in_entry:
            if self.N == self.__current_buffer_len:
                self.__current_buffer_len *= 2
                self.__data = <double **>realloc(
                    self.__data, self.__current_buffer_len*sizeof(double *))
            self.__data[self.N] = <double *>malloc(self.size)
            self.N += 1
            with gil:
                self.entries.add(s_str)
    
    cpdef void idxset(self, long idx, double [:] data) noexcept nogil:
        memcpy(<void*>self.__data[idx], <void*>&data[0], self.size)
    
    cdef void idxset_ptr(self, long idx, double * data) noexcept nogil:
        memcpy(<void*>self.__data[idx], <void*>data, self.size)

    cpdef void set(self, str s_str, double [:] data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset(idx, data)
    
    cdef void set_ptr(self, str s_str, double * data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset_ptr(idx, data)

    cpdef void unset(self, str s_str) noexcept nogil:
        cdef long idx, i
        with gil:
            idx = self.entries.get_index(s_str)
        if idx != -1:
            with gil:
                self.entries.remove(s_str)
            self.N -= 1
            free(self.__data[idx])
            for i in range(idx, self.N):
                self.__data[i] = self.__data[i+1]
            self.__data[self.N] = NULL

@cython.final
cdef class Vectors():
    def __cinit__(self, long ndata, long ndim, long N=3):
        cdef long i
        self.__current_buffer_len = N
        self.N = 0
        self.ndim = ndim
        self.nd2 = ndim*ndim
        self.ndata = ndata
        self.size = ndata*ndim*sizeof(double)
        self.entries = StrLists.__new__(StrLists)
        self.__data_agn_coord = <double **>malloc(self.__current_buffer_len*sizeof(double *))
        self.__data_obs_coord = <double **>malloc(self.__current_buffer_len*sizeof(double *))
        self.rotmat = <double *>calloc(self.nd2, sizeof(double))
        for i in range(self.ndim):
            self.rotmat[(self.ndim+1)*i] = 1

    def __dealloc__(self):
        cdef long i
        if self.N > 0:
            for i in range(self.N):
                if self.__data_agn_coord[i] != NULL:
                    free(self.__data_agn_coord[i])
                if self.__data_obs_coord[i] != NULL:
                    free(self.__data_obs_coord[i])
        if self.__data_agn_coord != NULL:
            free(self.__data_agn_coord)
        if self.__data_obs_coord != NULL:
            free(self.__data_obs_coord)
        if self.rotmat != NULL:
            free(self.rotmat)
    
    cpdef void rotate(self, double [:] rotmat) noexcept nogil: # vec.rotate(rotmat.ravel())
        cdef long i
        memcpy(<void*>self.rotmat, <void*>&rotmat[0], self.nd2*sizeof(double))
        for i in range(self.N):
            rotate_vector(
                self.ndata,
                self.ndim,
                self.rotmat,
                self.__data_agn_coord[i],
                self.__data_obs_coord[i],
            )
    
    cdef void rotate_ptr(self, double * rotmat) noexcept nogil: # vec.rotate(rotmat.ravel())
        cdef long i
        memcpy(<void*>self.rotmat, <void*>rotmat, self.nd2*sizeof(double))
        for i in range(self.N):
            rotate_vector(
                self.ndata,
                self.ndim,
                self.rotmat,
                self.__data_agn_coord[i],
                self.__data_obs_coord[i],
            )

    cpdef object get_agn(self, str s_str):
        cdef long i = self.entries.get_index(s_str)
        if i == -1:
            raise ValueError("No such entry")
        return numpy_dbl_2d(self.__data_agn_coord[i], self.ndata, self.ndim, False)
    
    cpdef object get_obs(self, str s_str):
        cdef long i = self.entries.get_index(s_str)
        if i == -1:
            raise ValueError("No such entry")
        return numpy_dbl_2d(self.__data_obs_coord[i], self.ndata, self.ndim, False)

    cpdef object get_agn_by_index(self, long i):
        return numpy_dbl_2d(self.__data_agn_coord[i], self.ndata, self.ndim, False)
    
    cpdef object get_obs_by_index(self, long i):
        return numpy_dbl_2d(self.__data_obs_coord[i], self.ndata, self.ndim, False)

    cpdef void allocate_entry(self, str s_str) noexcept nogil:
        cdef bint in_entry
        with gil:
            in_entry = self.entries.is_in(s_str)
        if not in_entry:
            if self.N == self.__current_buffer_len:
                self.__current_buffer_len *= 2
                self.__data_agn_coord = <double **>realloc(
                    self.__data_agn_coord, self.__current_buffer_len*sizeof(double *))
                self.__data_obs_coord = <double **>realloc(
                    self.__data_obs_coord, self.__current_buffer_len*sizeof(double *))    
            self.__data_agn_coord[self.N] = <double *>malloc(self.size)
            self.__data_obs_coord[self.N] = <double *>malloc(self.size)
            self.N += 1
            with gil:
                self.entries.add(s_str)

    cpdef void idxset_in_agn_coord(self, long idx, double [:,:] data) noexcept nogil:
        memcpy(<void*>self.__data_agn_coord[idx], <void*>&data[0,0], self.size)
        rotate_vector(self.ndata, self.ndim, self.rotmat, self.__data_agn_coord[idx], self.__data_obs_coord[idx])

    cpdef void idxset_in_obs_coord(self, long idx, double [:,:] data) noexcept nogil:
        memcpy(<void*>self.__data_obs_coord[idx], <void*>&data[0,0], self.size)
        rotate_vector_T(self.ndata, self.ndim, self.rotmat, self.__data_obs_coord[idx], self.__data_agn_coord[idx])
        
    cdef void idxset_in_agn_ptr(self, long idx, double * data) noexcept nogil:
        memcpy(<void*>self.__data_agn_coord[idx], <void*>data, self.size)
        rotate_vector(self.ndata, self.ndim, self.rotmat, self.__data_agn_coord[idx], self.__data_obs_coord[idx])

    cdef void idxset_in_obs_ptr(self, long idx, double * data) noexcept nogil:
        memcpy(<void*>self.__data_obs_coord[idx], <void*>data, self.size)
        rotate_vector_T(self.ndata, self.ndim, self.rotmat, self.__data_obs_coord[idx], self.__data_agn_coord[idx])    

    cpdef void set_in_agn_coord(self, str s_str, double [:,:] data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset_in_agn_coord(idx, data)

    cpdef void set_in_obs_coord(self, str s_str, double [:,:] data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset_in_obs_coord(idx, data)
        
    cdef void set_in_agn_ptr(self, str s_str, double * data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset_in_agn_ptr(idx, data)

    cdef void set_in_obs_ptr(self, str s_str, double * data) noexcept nogil:
        cdef bint in_entry
        cdef long idx
        with gil:
            in_entry = self.entries.is_in(s_str)
            if in_entry:
                idx = self.entries.get_index(s_str)
            else:
                idx = self.N
                self.allocate_entry(s_str)
        self.idxset_in_obs_ptr(idx, data)

    cpdef void unset(self, str s_str) noexcept nogil:
        cdef long idx, i
        with gil:
            idx = self.entries.get_index(s_str)
        if idx != -1:
            with gil:
                self.entries.remove(s_str)
            self.N -= 1
            free(self.__data_agn_coord[idx])
            free(self.__data_obs_coord[idx])
            for i in range(idx, self.N):
                self.__data_agn_coord[i] = self.__data_agn_coord[i+1]
                self.__data_obs_coord[i] = self.__data_obs_coord[i+1]
            self.__data_agn_coord[self.N] = NULL
            self.__data_obs_coord[self.N] = NULL