# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, exp, cos, sin, pow,
    acos, asin,
    fmin, fmax, fabs,
)

from .diskmesh cimport MeshType, DensityKeys, DensityIndexPair, packet_Mesh1D

from ..utils.types cimport (
    func_t_void_arr_auxarr
)
from libc.stdio cimport fprintf, stderr

from ..utils.numpy_interface cimport (
    numpy_dbl_1d, numpy_dbl_2d,
    numpy_lng_1d, numpy_lng_2d,
    numpy_ascontiguousarray, numpy_ravel_C,
)

from ..utils.algorithms cimport (
    find_min_max,
    binary_search_ptr_L, binary_search_ptr_R, 
    countingsort_uchar_ptr_subdivide, 
    count_each_items_uchar_ptr,
    assign_sorted_vector_inplace_dbl_ptr,
    assign_sorted_scalar_inplace_dbl_ptr,
    find_root, finite_difference_f, finite_difference_b,
)

from .transformables cimport (
    Scalars, Vectors, 
    rotate_vector, rotate_vector_T, 
    inner_prod_const, inner_prod_xyz,
)

from .diskmesh_auxil cimport (
    tri_annulus_connectivity, quad_annulus_connectivity,
    __assign_position_difference_vectors, __assign_triangle_area,
    __reorder_based_on_dblscalar, 
    _c_trim_mesh, 
    _c_reorder_scalar_after_trim, _c_reorder_vector_after_trim,
    make_obscuration_mask,
)

from ..geometry.diskgeom cimport (
    func_t_diskgeom, DiskGeometryModel,
    assign_azimuthal, compute_lamppost_distance,
)

from ..geometry.thin_goad cimport (
    assign_thin_goad_log_hexa, assign_thin_goad_log_rect,
    assign_thin_goad_lin_hexa, assign_thin_goad_lin_rect,
)

from ..geometry.slim_free cimport (
    assign_slim_free_log_hexa, assign_slim_free_log_rect,
    assign_slim_free_lin_hexa, assign_slim_free_lin_rect,
    slim_free_marginal_area,
)

from ..physics.kinematics cimport vel_field_disk_thick
from ..physics.response cimport (
    par_resp, ResponseModel, ReflectionModel, ThermalModel, ThermalDistanceScheme,
    respmodel_goad_pow, respmodel_goad_inv, respmodel_goad_invsq, 
    respmodel_T_M, respmodel_T_M_proj, respmodel_T_M_sym, respmodel_T_M_symproj, 
    respmodel_T_I, respmodel_T_I_proj, respmodel_T_I_sym, respmodel_T_I_symproj,
    ResponsePacket, freeResponsePacket,
    T_all, T_proj,
)
from ..physics.thermal cimport (
    ThermalPacket, Rgrav,
    Rsub_Kishimoto2007, Rsub_GRAVITY, Rsub_Nenkova,
    TPrep_Nenkova, par_nenkova, TProfile_Nenkova, 
    TPrep_NenkovaC, par_nenkovac, TProfile_NenkovaC, 
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

from .compute_tf cimport compute_tf_pdf, compute_tf_cdf

cdef double deg2rad = M_PI/180.
cdef double rad2deg = 180./M_PI

cdef long find_index_physics(long n, DensityIndexPair *index_to_physics, DensityKeys key) noexcept nogil:
    cdef long i
    for i in range(n):
        if index_to_physics[i].key == key:
            return index_to_physics[i].index
    return -1

cdef void free_packet_Mesh1D(packet_Mesh1D * packet) noexcept nogil:
    packet.__b_xyz        = NULL
    packet.__b_f_v_i      = NULL
    packet.__b_v_f_i      = NULL
    packet.__b_v_f_n      = NULL
    packet.__b_area       = NULL
    packet.__b_resp       = NULL
    packet.__b_emis       = NULL
    packet.__b_tau        = NULL
    packet.__b_precomp    = NULL
    packet.__b_average    = NULL
    packet.__b_cumulsum   = NULL
    packet.__b_mult       = NULL
    packet.__b_tau_f      = NULL
    packet.__b_idx_f      = NULL
    packet.lb_nf          = NULL
    packet.db_nf          = NULL
    packet.db_nfntfs      = NULL
    packet.mask_v         = NULL
    packet.mask_f         = NULL
    packet.mask_potential = NULL
    free(packet)

# Following is required to make cpdef functions nogil.
# this also renders this class uninheritable -- meaning it cannot form subclass (child class).
@cython.final
cdef class DiskMesh():
    def __cinit__(self, 
        long nrad, long nazm,
        str mesh_type='tri', 
        str geom_type='thin',#'slim',
        str resp_type='goad_pow',
        str thermal_distance='radial',
        # bint T_from_planar=True,
        bint track_velocity=False,
        bint store_topology=True,
        bint r_logbin=True,
        bint hexapack=True,
        long ntfs=1,
    ):
        # allocate memory and assign invariant topology (mesh fviectivity)
        # Don't assign other values
        self.nrad = nrad
        self.nazm = nazm 
        self.rotmat = <double *> malloc(9*sizeof(double))
        self.nv = (nrad+1)*nazm # it is always assumed.
        if mesh_type=='tri':
            self.mesh_type = MeshType.TRI
            self.nvpf = 3
            self.nfpv = 6
            self.nf = nrad*nazm*2
        elif mesh_type=='quad':
            self.mesh_type = MeshType.QUAD
            self.nvpf = 4
            self.nfpv = 4
            self.nf = nrad*nazm
        self._xyz0       = <double *> malloc((self.nv * 3) * sizeof(double))
        self._xyz        = <double *> malloc((self.nv * 3) * sizeof(double))
        self._normal0    = <double *> malloc((self.nv * 3) * sizeof(double))
        self._normal     = <double *> malloc((self.nv * 3) * sizeof(double))
        self._rho        = <double *> malloc( self.nv      * sizeof(double))
        self._R          = <double *> malloc( self.nv      * sizeof(double))
        self._RLamp      = <double *> malloc( self.nv      * sizeof(double))
        self._cell_areas = <double *> malloc( self.nf      * sizeof(double))
        
        self._f_v_i = <long *> malloc((self.nf * self.nvpf) * sizeof(long))
        self._v_f_i = <long *> malloc((self.nv * self.nfpv) * sizeof(long))
        self._v_f_n = <long *> malloc( self.nv              * sizeof(long))
    
        if store_topology:
            self.topology_is_stored = True
            self._f_v_i0 = <long *> malloc((self.nf * self.nvpf) * sizeof(long))
            self._v_f_i0 = <long *> malloc((self.nv * self.nfpv) * sizeof(long))
            self._v_f_n0 = <long *> malloc( self.nv              * sizeof(long))
        else:
            self.topology_is_stored = False
    
        # Numpy Interface
        self.f_v_i      = numpy_lng_2d(self._f_v_i,      self.nf, self.nvpf, False)
        self.v_f_i      = numpy_lng_2d(self._v_f_i,      self.nv, self.nfpv, False)
        self.v_f_n      = numpy_lng_1d(self._v_f_n,      self.nv,            False)
        self.xyz        = numpy_dbl_2d(self._xyz,        self.nv,         3, False)
        self.xyz0       = numpy_dbl_2d(self._xyz0,       self.nv,         3, False)
        self.normal     = numpy_dbl_2d(self._normal,     self.nv,         3, False)
        self.normal0    = numpy_dbl_2d(self._normal0,    self.nv,         3, False)
        self.rho        = numpy_dbl_1d(self._rho,        self.nv,            False)
        self.R          = numpy_dbl_1d(self._R,          self.nv,            False)
        self.cell_areas = numpy_dbl_1d(self._cell_areas, self.nf,            False)

        # Geometry Model Assignment
        if geom_type=='thin':
            if r_logbin:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LOG_HEXA
                    self.geom_func = assign_thin_goad_log_hexa
                else:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LOG_RECT
                    self.geom_func = assign_thin_goad_log_rect
            else:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LIN_HEXA
                    self.geom_func = assign_thin_goad_lin_hexa
                else:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LIN_RECT
                    self.geom_func = assign_thin_goad_lin_rect
        elif geom_type=='slim':
            if r_logbin:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.SLIM_FREE_LOG_HEXA
                    self.geom_func = assign_slim_free_log_hexa
                else:
                    self.geom_mod = DiskGeometryModel.SLIM_FREE_LOG_RECT
                    self.geom_func = assign_slim_free_log_rect
            else:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.SLIM_FREE_LIN_HEXA
                    self.geom_func = assign_slim_free_lin_hexa
                else:
                    self.geom_mod = DiskGeometryModel.SLIM_FREE_LIN_RECT
                    self.geom_func = assign_slim_free_lin_rect
        else: # Fall back to thin
            if r_logbin:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LOG_HEXA
                    self.geom_func = assign_thin_goad_log_hexa
                else:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LOG_RECT
                    self.geom_func = assign_thin_goad_log_rect
            else:
                if hexapack:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LIN_HEXA
                    self.geom_func = assign_thin_goad_lin_hexa
                else:
                    self.geom_mod = DiskGeometryModel.THIN_GOAD_LIN_RECT
                    self.geom_func = assign_thin_goad_lin_rect

        ##############################
        ### Vectors and Scalars are placeholders; they do not allocate memory until each entry is named.
        ###   But if don't reserve enough spaces, "realloc" calls will be made, which will slow down the code
        ###   and may mess up with memory. This is the reason I start with exact number of entries.
        ### If the user adds one or two more entries interactively, then it wouldn't be performance-critical,
        ###   so, it is not allocated more than it needs. 
        ### Also, it "doubles" the entry, not adds, so it never should have 0 entries.
        ##############################
        # Density variables; only transformed not conserved. (e.g., velocity, intensity)
        # Assign them only at points not at centers.
        self.n_density_dict = 3
        self.density_dict = <DensityIndexPair *> malloc(self.n_density_dict * sizeof(DensityIndexPair))
        if self.density_dict == NULL:
            raise MemoryError("Memory allocation failed for density_dict.")
        self.ScalDens = Scalars.__new__(Scalars, self.nv,    2) # length nv, 1 dim, 2 entries
        self.VecDens  = Vectors.__new__(Vectors, self.nv, 3, 1) # length nv, 3 dim, 1 entry
        # Integration variables; should be conserved upon transformation. (e.g., mass, total power)
        # Assign them at centers, not at points.
        self.ScalIntg = Scalars.__new__(Scalars, self.nf,    1) # length nf, 1 dim, 1 entry
        self.VecIntg  = Vectors.__new__(Vectors, self.nf, 3, 1) # length nf, 3 dim, 1 entry
        ### Memory allocation happens here.
        self.ScalDens.allocate_entry('tau')               # Time lag
        # self.ScalDens.allocate_entry('Resp_raw')          # Response before accounting for angle of incidence
        # self.ScalDens.allocate_entry('Response')          # Response after  accounting for angle of incidence
        self.ScalDens.allocate_entry('bowl_edge_forward') # Function for computing self-obscuration
        if track_velocity:
            self.track_velocity = True
            self.VecDens.allocate_entry('velocity') # Velocity vector field
        else:
            self.track_velocity = False
        self.assign_dict_indices()

        # Allocate Physics
        # Responses
        # Perhaps follow the example of stats.pyx. 
        # At least, make sure _resp_raw is single-depth pointer, with all elements pointing towards to the same vertex are consecutive.
        self.ntfs = ntfs
        self._resp_raw   = <double *> malloc((self.ntfs * self.nv) * sizeof(double))
        self._response   = <double *> malloc((self.ntfs * self.nv) * sizeof(double))
        self._emis_raw   = <double *> malloc((self.ntfs * self.nv) * sizeof(double))
        self._emission   = <double *> malloc((self.ntfs * self.nv) * sizeof(double))
        self.resp_raw = numpy_dbl_2d(self._resp_raw, self.nv, self.ntfs, False)
        self.response = numpy_dbl_2d(self._response, self.nv, self.ntfs, False)
        self.emis_raw = numpy_dbl_2d(self._emis_raw, self.nv, self.ntfs, False)
        self.emission = numpy_dbl_2d(self._emission, self.nv, self.ntfs, False)
        # If there are more than one response, this would change into <ResponsePacket **>.
        # See comments here and pxd file
        self._Rpacket         = <ResponsePacket *> malloc(sizeof(ResponsePacket))
        if self._Rpacket == NULL:
            raise MemoryError("Memory allocation failed for _Rpacket.")
        # Initialize all pointers to NULL for safe cleanup
        self._Rpacket.band = NULL
        self._Rpacket.x_rb = NULL
        self._Rpacket.coef_r = NULL
        self._Rpacket.coef_e = NULL
        self._Rpacket.Tpacket = NULL
        self._Rpacket.pars = NULL
        self._Rpacket.nband   = self.ntfs # if there are more than one _Rpacket, nband<ntfs
        self._Rpacket.offset  = 0 # if there are more than one _Rpacket, offset may be >0
        self._Rpacket.nvr     = self.nrad+1
        self._Rpacket.nva     = self.nazm
        self._Rpacket.azm_sym = True # False is implemented, but not used right now.
        self._Rpacket.pars    = <par_resp *>malloc(sizeof(par_resp))

        if 'goad' not in resp_type.lower():
            if '_m' in resp_type.lower():
                self.resp_mod = ResponseModel.RESP_THERMAL_M
                self._Rpacket.band = <double *>malloc(self.ntfs*sizeof(double))
                self._Rpacket.n_rb = -1
            else:
                self.resp_mod = ResponseModel.RESP_THERMAL_I
                self._Rpacket.band = NULL
                self._Rpacket.n_rb = 0
            self._Rpacket.x_rb = NULL
            self._Rpacket.coef_r = NULL
            self._Rpacket.coef_e = NULL
            self._Rpacket.Tpacket = <ThermalPacket *>malloc(sizeof(ThermalPacket))
            if 'nenkovac' in resp_type.lower():
                self._Rpacket.pars.mod_therm = ThermalModel.T_NENKOVAC
                self._Rpacket.Tpacket.par_thermal = <void *>malloc(sizeof(par_nenkovac))
            elif 'nenkova' in resp_type.lower():
                self._Rpacket.pars.mod_therm = ThermalModel.T_NENKOVA
                self._Rpacket.Tpacket.par_thermal = <void *>malloc(sizeof(par_nenkova))
            else: # Fallback.
                self._Rpacket.pars.mod_therm = ThermalModel.T_NENKOVA
                # Following choice will be implemented later.
                self._Rpacket.Tpacket.par_thermal = <void *>malloc(sizeof(par_nenkova))
        else: # FIX THIS since this causes deallocation segfault.
            # self._Rpacket = <void *> NULL
            self._Rpacket.pars.mod_therm = ThermalModel.T_ISOTHERMAL
            self._Rpacket.band = NULL
            self._Rpacket.x_rb = NULL
            self._Rpacket.coef_r = NULL
            self._Rpacket.coef_e = NULL
            self._Rpacket.Tpacket = NULL
            # Following choice will be implemented later.
            self.resp_mod = ResponseModel.RESP_GOAD_POW
        self.resp_data = <void **> malloc(6*sizeof(void*))
        if   thermal_distance=='radial':
            self.thermal_dist = ThermalDistanceScheme.DIST_RADIAL
        elif thermal_distance=='planar':
            self.thermal_dist = ThermalDistanceScheme.DIST_PLANAR
        elif thermal_distance=='lamppost':
            self.thermal_dist = ThermalDistanceScheme.DIST_LAMPPOST
        else: # fallback to radial
            self.thermal_dist = ThermalDistanceScheme.DIST_RADIAL
        # Following choice will be implemented later.
        self._Rpacket.pars.mod_resp = self.resp_mod
        self._Rpacket.pars.mod_inci = ReflectionModel.REF_LAMBERTIAN
        self._Rpacket.pars.mod_refl = ReflectionModel.REF_LAMBERTIAN

        # Allocate Primitives
        self._rad1d = <double *> malloc((self.nrad+1)*sizeof(double))
        self._azm1d = <double *> malloc( self.nazm   *sizeof(double))
        self._z1d   = <double *> malloc((self.nrad+1)*sizeof(double))

        # Allocate Masks
        self.mask_v  = <unsigned char *> malloc(self.nv*sizeof(unsigned char))
        self.mask_f  = <unsigned char *> malloc(self.nf*sizeof(unsigned char))
        self.mask_potential = <double *> malloc(self.nv*sizeof(double))

        # Allocate Buffers
        ## Generic
        self.lb_nv = <long   *> malloc(self.nv * sizeof(long  ))
        self.lb_nf = <long   *> malloc(self.nf * sizeof(long  ))
        self.db_nv = <double *> malloc(self.nv * sizeof(double))
        self.db_nf = <double *> malloc(self.nf * sizeof(double))
        ## Specific
        self.db_nv3 = <double *> malloc(self.nv*3*sizeof(double))
        self.db_nvntfs = <double *> malloc(self.nv*self.ntfs*sizeof(double))
        self.db_nfntfs = <double *> malloc(self.nf*self.ntfs*sizeof(double))
        self.db_ntfs = <double *> malloc(self.ntfs*sizeof(double))
        # self.db_nsrc = <double *> malloc((self.nrad+1)*sizeof(double))
        if self.mesh_type==MeshType.TRI:
            self.db_area_vector_pairs = <double *> malloc(self.nf*9*sizeof(double))
        elif self.mesh_type==MeshType.QUAD:
            self.db_area_vector_pairs = <double *> malloc(self.nf*6*sizeof(double))
        else: # Default
            self.db_area_vector_pairs = <double *> malloc(self.nf*9*sizeof(double))
        ## Integration   (self.ntfs * self.nv) * sizeof(double)
        self.__b_xyz   = <double *> malloc(self.nv * 3         * sizeof(double))
        self.__b_f_v_i = <long   *> malloc(self.nf * self.nvpf * sizeof(long  ))
        self.__b_v_f_i = <long   *> malloc(self.nv * self.nfpv * sizeof(long  ))
        self.__b_v_f_n = <long   *> malloc(self.nv             * sizeof(long  ))
        self.__b_area  = <double *> malloc(self.nf             * sizeof(double))
        self.__b_idx_v = <long   *> malloc(self.nv             * sizeof(long  ))
        self.__b_idx_f = <long   *> malloc(self.nf             * sizeof(long  ))
        self.__b_nip   = <long   *> malloc(self.nv             * sizeof(long  ))
        self.__b_iip   = <long   *> malloc(self.nv * 3         * sizeof(long  ))
        self.__b_iw    = <double *> malloc(self.nv * 2         * sizeof(double))
        self.__b_tau   = <double *> malloc(self.nv             * sizeof(double))
        self.__b_resp  = <double *> malloc(self.nv * self.ntfs * sizeof(double))
        self.__b_emis  = <double *> malloc(self.nv * self.ntfs * sizeof(double))
        self.__b_precomp  = <double *> malloc(self.nf * self.ntfs * sizeof(double))
        self.__b_average  = <double *> malloc(self.nf * self.ntfs * sizeof(double))
        self.__b_cumulsum = <double *> malloc(self.nf * self.ntfs * sizeof(double))
        self.__b_mult     = <double *> malloc(self.ntfs * sizeof(double))
        self.__b_tau_f    = <double *> malloc(self.nf * sizeof(double))
        
        if (
            self._xyz0 == NULL or self._xyz == NULL or self._normal0 == NULL or self._normal == NULL 
            or self._rho == NULL or self._R == NULL or self._cell_areas == NULL or self._f_v_i == NULL 
            or self._v_f_i == NULL or self._v_f_n == NULL or self.rotmat == NULL 
            or self._rad1d == NULL or self._azm1d == NULL or self._z1d == NULL
            or self.mask_v == NULL or self.mask_f == NULL or self.mask_potential == NULL
            or self.lb_nv == NULL or self.lb_nf == NULL #or self.db_nsrc == NULL
            or self.db_nv == NULL or self.db_nf == NULL or self.db_nv3 == NULL 
            or self.db_nvntfs == NULL or self.db_nfntfs == NULL or self.db_ntfs == NULL or self._RLamp == NULL
            or self.db_area_vector_pairs == NULL
            or self._resp_raw == NULL or self._response == NULL or self._emis_raw == NULL or self._emission == NULL
            or self._Rpacket == NULL
            or self.__b_xyz == NULL or self.__b_f_v_i == NULL or self.__b_v_f_i == NULL or self.__b_v_f_n == NULL
            or self.__b_area == NULL or self.__b_idx_v == NULL or self.__b_idx_f == NULL
            or self.__b_nip == NULL or self.__b_iip == NULL or self.__b_iw == NULL 
            or self.__b_tau == NULL or self.__b_resp == NULL or self.__b_emis == NULL
            or self.__b_precomp == NULL or self.__b_average == NULL or self.__b_mult == NULL
            or self.__b_cumulsum == NULL or self.__b_tau_f == NULL
            or self.density_dict == NULL or self.resp_data == NULL
            or (self.topology_is_stored and (
                self._f_v_i0 == NULL or self._v_f_i0 == NULL or self._v_f_n0 == NULL
            ))
        ):
            raise MemoryError("Memory allocation failed.")
        
        # Assign topology
        self.thread_mesh()
    
    def __dealloc__(self):
        if self._xyz0                != NULL: free(self._xyz0);
        if self._xyz                 != NULL: free(self._xyz);
        if self._normal0             != NULL: free(self._normal0);
        if self._normal              != NULL: free(self._normal);
        if self._rho                 != NULL: free(self._rho);
        if self._R                   != NULL: free(self._R);
        if self._cell_areas          != NULL: free(self._cell_areas);
        if self._f_v_i               != NULL: free(self._f_v_i);
        if self._v_f_i               != NULL: free(self._v_f_i);
        if self._v_f_n               != NULL: free(self._v_f_n);
        if self.rotmat               != NULL: free(self.rotmat);
        if self._rad1d               != NULL: free(self._rad1d);
        if self._azm1d               != NULL: free(self._azm1d);
        if self._z1d                 != NULL: free(self._z1d);
        if self.mask_v               != NULL: free(self.mask_v);
        if self.mask_f               != NULL: free(self.mask_f);
        if self.mask_potential       != NULL: free(self.mask_potential);
        if self.lb_nv                != NULL: free(self.lb_nv);
        if self.lb_nf                != NULL: free(self.lb_nf);
        if self.db_nv                != NULL: free(self.db_nv);
        if self.db_nf                != NULL: free(self.db_nf);
        if self.db_nv3               != NULL: free(self.db_nv3);
        if self.db_nvntfs            != NULL: free(self.db_nvntfs);
        if self.db_nfntfs            != NULL: free(self.db_nfntfs);
        if self.db_ntfs              != NULL: free(self.db_ntfs);
        if self._RLamp               != NULL: free(self._RLamp);
        if self.db_area_vector_pairs != NULL: free(self.db_area_vector_pairs);
        if self._resp_raw            != NULL: free(self._resp_raw);
        if self._response            != NULL: free(self._response);
        if self._emis_raw            != NULL: free(self._emis_raw);
        if self._emission            != NULL: free(self._emission);
        if self._Rpacket             != NULL: freeResponsePacket(self._Rpacket);
        if self.__b_xyz              != NULL: free(self.__b_xyz);
        if self.__b_f_v_i            != NULL: free(self.__b_f_v_i);
        if self.__b_v_f_i            != NULL: free(self.__b_v_f_i);
        if self.__b_v_f_n            != NULL: free(self.__b_v_f_n);
        if self.__b_area             != NULL: free(self.__b_area);
        if self.__b_idx_v            != NULL: free(self.__b_idx_v);
        if self.__b_idx_f            != NULL: free(self.__b_idx_f);
        if self.__b_nip              != NULL: free(self.__b_nip);
        if self.__b_iip              != NULL: free(self.__b_iip);
        if self.__b_iw               != NULL: free(self.__b_iw);
        if self.__b_tau              != NULL: free(self.__b_tau);
        if self.__b_resp             != NULL: free(self.__b_resp);
        if self.__b_emis             != NULL: free(self.__b_emis);
        if self.__b_precomp          != NULL: free(self.__b_precomp);
        if self.__b_average          != NULL: free(self.__b_average);
        if self.__b_mult             != NULL: free(self.__b_mult);
        if self.__b_cumulsum         != NULL: free(self.__b_cumulsum);
        if self.__b_tau_f            != NULL: free(self.__b_tau_f);
        if self.density_dict         != NULL: free(self.density_dict);
        if self.resp_data            != NULL: free(self.resp_data);
        if self.topology_is_stored:
            if self._f_v_i0 != NULL: free(self._f_v_i0);
            if self._v_f_i0 != NULL: free(self._v_f_i0);
            if self._v_f_n0 != NULL: free(self._v_f_n0);

    cpdef void thread_mesh(self,) noexcept nogil:
        if self.mesh_type==MeshType.TRI:
            # Connect the points
            tri_annulus_connectivity(
                self.nrad, self.nazm, self._f_v_i, self._v_f_i, self._v_f_n,
            )
        elif self.mesh_type==MeshType.QUAD:
            # Connect the points
            quad_annulus_connectivity(
                self.nrad, self.nazm, self._f_v_i, self._v_f_i, self._v_f_n,
            )
        if self.topology_is_stored:
            memcpy(<void*>self._f_v_i0, <void*>self._f_v_i, (self.nf*self.nvpf)*sizeof(long))
            memcpy(<void*>self._v_f_i0, <void*>self._v_f_i, (self.nv*self.nfpv)*sizeof(long))
            memcpy(<void*>self._v_f_n0, <void*>self._v_f_n,  self.nv           *sizeof(long))

    cpdef void assign_mesh(self, 
        double R_in, double R_out,
        double c_f, 
        double curv, 
        double log1pz=0., # ln(1 + z)
        double logLbol=44.09935780,
        double logMass=7.,
        double H_lamp=20.,
        double pindex=1.,
        double LampEff=0.1,
        bint force_proj=True,
    ) noexcept nogil:
        # This is to make default value. There should be a separate function to refresh mesh.
        cdef long i, j
        # Fundamental (input) gemoetric variables
        self.R_in  = R_in
        self.R_out = R_out
        self.c_f   = fabs(c_f)
        self.curv  = curv
        # Fundamental physical variables
        self.log1pz  = log1pz
        self.logLbol = logLbol
        self.logMass = logMass
        self.Rg      = Rgrav(self.logMass)
        self.H_lamp  = H_lamp
        self.H_lamp_ltday = self.H_lamp*self.Rg
        self.pindex  = pindex
        self.LampEff = LampEff
        # 
        self._Rpacket.pars.log1pz  = &self.log1pz
        self._Rpacket.pars.logLbol = &self.logLbol
        self._Rpacket.pars.H_lamp  = &self.H_lamp_ltday
        self._Rpacket.pars.logMass = &self.logMass
        self._Rpacket.pars.pindex  = &self.pindex
        self._Rpacket.pars.curv    = &self.curv
        self._Rpacket.pars.c_f     = &self.c_f
        self._Rpacket.pars.LampEff = &self.LampEff

        # Computing Primitives
        assign_azimuthal(self.nazm, self._azm1d) # This doesn't need to be run in refreshing runs.
        self.geom_func(
            &self.rho_out, self._rho, self._R, self._xyz0, self._normal0,
            self.nazm, self.nrad, self.nv, self.R_out, self.R_in, self.c_f, self.curv,
            self._azm1d, self._rad1d, self._z1d, NULL
        )
        compute_lamppost_distance(self.nv, self._RLamp, self._xyz0, self._rho, self.H_lamp_ltday)
        self.reset_rotation()
        # Given band data are not usually initialized at this point,
        # self.compute_pointwise_geometry()
        self.compute_cellwise_geometry()

        # Assigning Physics
        # if self.T_from_planar:
        #     self.resp_data[0] = <void*> self._rho
        # else:
        #     self.resp_data[0] = <void*> self._R
        if   self.thermal_dist==ThermalDistanceScheme.DIST_RADIAL:
            self.resp_data[0] = <void*> self._R
        elif self.thermal_dist==ThermalDistanceScheme.DIST_PLANAR:
            self.resp_data[0] = <void*> self._rho
        elif self.thermal_dist==ThermalDistanceScheme.DIST_LAMPPOST:
            self.resp_data[0] = <void*> self._RLamp
        else: # Fallback to DIST_RADIAL
            self.resp_data[0] = <void*> self._R
        self.resp_data[1] = <void*> self._normal
        self.resp_data[2] = <void*> self._xyz
        self.resp_data[3] = <void*> self._R # Should not use this, this is not mixed?
        self.resp_data[4] = <void*> self._RLamp
        if self._Rpacket.Tpacket != NULL:
            # self.resp_data[5] = <void*> self.ScalDens.__data[self.get_density_idx(DensityKeys.TEMPERATURE)]
            # self.resp_data[5] = <void*> self.db_nsrc
            self.resp_data[5] = <void*> self.db_nv

        # Selecting Resp Function
        if   self.resp_mod == ResponseModel.RESP_GOAD_INV:
            self.resp_func = respmodel_goad_inv
            self.fRsub = Rsub_Nenkova
        elif self.resp_mod == ResponseModel.RESP_GOAD_POW:
            self.resp_func = respmodel_goad_pow
            self.fRsub = Rsub_Nenkova
        elif self.resp_mod == ResponseModel.RESP_GOAD_INVSQ:
            self.resp_func = respmodel_goad_invsq
            self.fRsub = Rsub_Nenkova
        elif self.resp_mod == ResponseModel.RESP_THERMAL_M or ResponseModel.RESP_THERMAL_I:
            if self._Rpacket.azm_sym:
                if self.resp_mod == ResponseModel.RESP_THERMAL_M:
                    if self.H_lamp>1 or force_proj:
                        self.resp_func = respmodel_T_M_symproj
                    else:
                        self.H_lamp=0
                        self.resp_func = respmodel_T_M_sym
                else:
                    if self.H_lamp>1 or force_proj:
                        self.resp_func = respmodel_T_I_symproj
                    else:
                        self.H_lamp=0
                        self.resp_func = respmodel_T_I_sym
            else:
                if self.resp_mod == ResponseModel.RESP_THERMAL_M:
                    if self.H_lamp>1 or force_proj:
                        self.resp_func = respmodel_T_M_proj
                    else:
                        self.H_lamp=0
                        self.resp_func = respmodel_T_M
                else:
                    if self.H_lamp>1 or force_proj:
                        self.resp_func = respmodel_T_I_proj
                    else:
                        self.H_lamp=0
                        self.resp_func = respmodel_T_I
            if self._Rpacket.pars.mod_therm == ThermalModel.T_NENKOVAC:
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).logLbol = &self.logLbol
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).Rg      = &self.Rg
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).r_ISCO   = 6.
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).f_cutoff = 1.0
                self._Rpacket.Tpacket.T = TProfile_NenkovaC
                self._Rpacket.Tpacket.Tprep = TPrep_NenkovaC
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).fRsub = Rsub_Nenkova
                self.fRsub = Rsub_Nenkova
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).adust = 0.05
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).Tsub  = 1500.
                (<par_nenkovac*>(self._Rpacket.Tpacket.par_thermal)).BCvLv = 10.
                self._Rpacket.Tpacket.Tprep(self._Rpacket.Tpacket.par_thermal) # FIX THIS!
            elif self._Rpacket.pars.mod_therm == ThermalModel.T_NENKOVA:
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).logLbol = &self.logLbol
                self._Rpacket.Tpacket.T = TProfile_Nenkova
                self._Rpacket.Tpacket.Tprep = TPrep_Nenkova
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).fRsub = Rsub_Nenkova
                self.fRsub = Rsub_Nenkova
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).adust = 0.05
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).Tsub  = 1500.
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).BCvLv = 10.
                self._Rpacket.Tpacket.Tprep(self._Rpacket.Tpacket.par_thermal) # FIX THIS!
            else: # Fallback to T_NENKOVA
                self._Rpacket.pars.mod_therm = ThermalModel.T_NENKOVA
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).logLbol = &self.logLbol
                self._Rpacket.Tpacket.T = TProfile_Nenkova
                self._Rpacket.Tpacket.Tprep = TPrep_Nenkova
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).fRsub = Rsub_Nenkova
                self.fRsub = Rsub_Nenkova
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).adust = 0.05
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).Tsub  = 1500.
                (<par_nenkova*>(self._Rpacket.Tpacket.par_thermal)).BCvLv = 10.
                self._Rpacket.Tpacket.Tprep(self._Rpacket.Tpacket.par_thermal) # FIX THIS!
        else: # Fallback to RESP_GOAD_INV
            self.resp_mod  = ResponseModel.RESP_GOAD_INV
            self.resp_func = respmodel_goad_inv
            self.fRsub = Rsub_Nenkova

    cdef void reset_rotation(self) noexcept nogil:
        cdef long i, j
        # Reset the rotation matrix
        self.incl = 0.
        for i in range(3):
            self.rotmat[4*i] = 1
            for j in range(i):
                self.rotmat[3*i+j] = 0
                self.rotmat[3*j+i] = 0
        # Copying AGN coordinates to OBS coordinates
        memcpy(<void*> self._xyz,    <void*> self._xyz0,    self.nv*3*sizeof(double))
        memcpy(<void*> self._normal, <void*> self._normal0, self.nv*3*sizeof(double))

    cdef void assign_band_M(self, double * wave) nogil:
        # long isrc=0 may be added to the function signature.
        # cdef long offset = self._Rpacket.offset # Not necessary
        if self._Rpacket.band == NULL:
            raise MemoryError("Band array not allocated.\nThis is a memory allocation bug. Restarting the kernel may help.\nIf the problem persists, please report.")
        memcpy(<void *>(self._Rpacket.band), <void *>wave, (self._Rpacket.nband)*sizeof(double))

    cdef void assign_band_I(self, double * x_rb, double * coef_r, double * coef_e, long n_rb) nogil:
        # long isrc=0 may be added to the function signature.
        # cdef long offset = self._Rpacket.offset # Not necessary
        self._Rpacket.n_rb = n_rb
        if self._Rpacket.x_rb == NULL:
            self._Rpacket.x_rb = <double *>malloc(n_rb*sizeof(double))
        if self._Rpacket.coef_r == NULL:
            self._Rpacket.coef_r = <double *>malloc(4*self._Rpacket.nband*(n_rb+1)*sizeof(double))
        if self._Rpacket.coef_e == NULL:
            self._Rpacket.coef_e = <double *>malloc(4*self._Rpacket.nband*(n_rb+1)*sizeof(double))
        if self._Rpacket.x_rb == NULL or self._Rpacket.coef_r == NULL or self._Rpacket.coef_e == NULL:
            if self._Rpacket.x_rb   != NULL: 
                free(self._Rpacket.x_rb  ); self._Rpacket.x_rb   = NULL
            if self._Rpacket.coef_r != NULL: 
                free(self._Rpacket.coef_r); self._Rpacket.coef_r = NULL
            if self._Rpacket.coef_e != NULL: 
                free(self._Rpacket.coef_e); self._Rpacket.coef_e = NULL
            raise MemoryError("Memory allocation failed for the interpolation data.")
        memcpy(<void *>(self._Rpacket.x_rb), <void *>x_rb, n_rb*sizeof(double))
        memcpy(<void *>(self._Rpacket.coef_r), <void *>coef_r, 4*self._Rpacket.nband*(n_rb+1)*sizeof(double))
        memcpy(<void *>(self._Rpacket.coef_e), <void *>coef_e, 4*self._Rpacket.nband*(n_rb+1)*sizeof(double))

    cpdef void assign_band(self, object x, object coef_r=None, object coef_e=None):
        cdef long n
        cdef double [:] xmv
        cdef double [:] cmv_r
        cdef double [:] cmv_e
        cdef double [:,:,:] cmv_r_3d
        cdef double [:,:,:] cmv_e_3d
        if self.resp_mod == ResponseModel.RESP_THERMAL_I:
            if coef_r is None or coef_e is None:
                raise ValueError("coef_r and coef_e must be provided for the interpolation data.")
            xmv = numpy_ravel_C(x)
            n = xmv.size
            cmv_r = numpy_ravel_C(coef_r)
            cmv_e = numpy_ravel_C(coef_e)
            if cmv_e.size!=cmv_r.size:
                raise ValueError("coef_e and coef_r must have the same size.")
            if cmv_r.size!= 4*self._Rpacket.nband*(n+1):
                cmv_r_3d = numpy_ascontiguousarray(coef_r)
                cmv_e_3d = numpy_ascontiguousarray(coef_e)
                if cmv_r_3d.shape[2] != 4 or cmv_r_3d.shape[0] != n+1:
                    raise ValueError("coef_r must be a 3D array with shape (n_rb+1, nband, 4), \n            with the second dimension at least the number of bands in the model.")
                elif cmv_r_3d.shape[1] < self._Rpacket.nband:
                    raise ValueError("The number of bands in the provided coef_r is smaller than the number of bands in the model.")
                # else:
                #     cmv_r = cmv_r_3d.ravel()
                elif cmv_r_3d.shape[0]!=cmv_e_3d.shape[0] or cmv_r_3d.shape[1]!=cmv_e_3d.shape[1]:
                    raise ValueError("coef_r and coef_e must have the same shape.")
                else:
                    fprintf(stderr,
                        "Warning (Number of Bands): coef_r has %d bands, but only first %d bands were used.\n",
                        cmv_r_3d.shape[1], self._Rpacket.nband
                    )
                    cmv_r = numpy_ravel_C(cmv_r_3d[:, :self._Rpacket.nband, :])
                    cmv_e = numpy_ravel_C(cmv_e_3d[:, :self._Rpacket.nband, :])
            with nogil:
                self.assign_band_I(&xmv[0], &cmv_r[0], &cmv_e[0], n)
        elif self.resp_mod == ResponseModel.RESP_THERMAL_M:
            xmv = numpy_ravel_C(x)
            with nogil:
                self.assign_band_M(&xmv[0])
        else:
            raise ValueError("Response model not supported for this function.")
        with nogil:
            # if self.topology_is_stored:
            #     memcpy(<void*>self._f_v_i, <void*>self._f_v_i0, (self.nf*self.nvpf)*sizeof(long))
            #     memcpy(<void*>self._v_f_i, <void*>self._v_f_i0, (self.nv*self.nfpv)*sizeof(long))
            #     memcpy(<void*>self._v_f_n, <void*>self._v_f_n0,  self.nv           *sizeof(long))
            # else:
            #     self.thread_mesh()
            # self.rotate_y(0.)
            self._update_geometry()
            self.compute_pointwise_physics()
            self.compute_cellwise_physics()

    cpdef void update_mesh(self, 
        double R_in, double R_out,
        double c_f, # covering factor
        double curv, 
        double log1pz=0., # ln(1 + z)
        double logLbol=44.09935780,
        double logMass=7.,
        double H_lamp=20.,
        double pindex=1.,
        double LampEff=0.1,
    ) noexcept nogil:
        self.curv  = curv
        self.c_f   = fabs(c_f)
        self.R_in  = R_in
        self.R_out = R_out
        self.log1pz  = log1pz
        self.logLbol = logLbol
        self.logMass = logMass
        self.Rg      = Rgrav(self.logMass)
        self.H_lamp  = H_lamp
        self.H_lamp_ltday = self.H_lamp*self.Rg
        self.pindex  = pindex
        self.LampEff = LampEff
        self._update_geometry()
        self._update_physics()

    cpdef void update_geometry(self, 
        double R_in, double R_out,
        double c_f, # covering factor
        double curv, 
        double H_lamp,
    ) noexcept nogil:
        self.curv  = curv
        self.c_f   = fabs(c_f)
        self.R_in  = R_in
        self.R_out = R_out
        self.H_lamp = H_lamp
        self.H_lamp_ltday = self.H_lamp*self.Rg
        self._update_geometry()

    cpdef void update_physics(self, 
        double log1pz=0., # ln(1 + z)
        double logLbol=44.09935780,
        double logMass=7.,
        double H_lamp=20.,
        double pindex=1.,
        double LampEff=0.1,
    ) noexcept nogil:
        self.log1pz  = log1pz
        self.logLbol = logLbol
        self.logMass = logMass
        self.Rg      = Rgrav(self.logMass)
        self.H_lamp  = H_lamp
        self.H_lamp_ltday = self.H_lamp*self.Rg
        self.pindex  = pindex
        self.LampEff = LampEff
        self._update_physics()

    cdef void _update_geometry(self,) noexcept nogil:
        cdef long i, j
        # Previous computation must have mixed vertices index.
        if self.topology_is_stored:
            memcpy(<void*>self._f_v_i, <void*>self._f_v_i0, (self.nf*self.nvpf)*sizeof(long))
            memcpy(<void*>self._v_f_i, <void*>self._v_f_i0, (self.nv*self.nfpv)*sizeof(long))
            memcpy(<void*>self._v_f_n, <void*>self._v_f_n0,  self.nv           *sizeof(long))
        else:
            self.thread_mesh()
        # Assigning coordinates
        self.geom_func(
            &self.rho_out, self._rho, self._R, self._xyz0, self._normal0,
            self.nazm, self.nrad, self.nv, self.R_out, self.R_in, self.c_f, self.curv, 
            self._azm1d, self._rad1d, self._z1d, NULL
        )
        compute_lamppost_distance(self.nv, self._RLamp, self._xyz0, self._rho, self.H_lamp_ltday)
        self.reset_rotation()
        # Computing other geometric & physical properties
        # self.compute_pointwise_geometry()
        self.compute_cellwise_geometry()

    cdef void _update_physics(self,) noexcept nogil:
        self.compute_pointwise_physics()
        self.compute_cellwise_physics()

    cpdef void assign_dict_indices(self) noexcept:
        # When to run this? 
        # If additional variables are added, this should be run *manually*, not by default.
        cdef long i
        for i, (key, strkey) in enumerate(zip(
            # [DensityKeys.TIME_LAG, DensityKeys.RESP_RAW, DensityKeys.RESPONSE, DensityKeys.OBSC_POTENTIAL], 
            # ['tau','Resp_raw','Response','bowl_edge_forward'],
            [DensityKeys.TIME_LAG, DensityKeys.OBSC_POTENTIAL], 
            ['tau','bowl_edge_forward'],
        )):
            self.density_dict[i].key = key
            self.density_dict[i].index = self.ScalDens.entries.get_index(strkey)
        self.density_dict[self.n_density_dict-1].key = DensityKeys.VELOCITY
        self.density_dict[self.n_density_dict-1].index = self.VecDens.entries.get_index('velocity')

    cpdef long get_density_idx(self, DensityKeys key) noexcept nogil:
        return find_index_physics(self.n_density_dict, self.density_dict, key)

    cdef void assign_velocity(self) noexcept nogil:
        vel_field_disk_thick(self.nv, self.logMass, self._xyz0, self._R, self.db_nv3)
        self.VecDens.idxset_in_agn_ptr(self.get_density_idx(DensityKeys.VELOCITY), self.db_nv3)
    
    cdef void assign_response_int(self) noexcept nogil:
        self.resp_func(
            self.nv, self.ntfs, <void *> self._resp_raw, <void *> self._emis_raw, 
            self.resp_data, self._Rpacket,
        )
    
    cpdef void assign_physics(self) noexcept nogil:
        if self.track_velocity:
            self.assign_velocity()
        self.assign_response_int() # Rename to something like assign_surface or assign_optics?

    # cpdef void compute_pointwise_properties(self) noexcept nogil:
    #     self.compute_pointwise_geometry()
    #     self.compute_pointwise_physics()
        
    # cpdef void compute_cellwise_properties(self,) noexcept nogil:
    #     self.compute_cellwise_geometry()
    #     self.compute_cellwise_physics()

    # cpdef void compute_pointwise_geometry(self) noexcept nogil:
    #     self.compute_normals()
    #     memcpy(<void*> self._normal, <void*> self._normal0, self.nv*3*sizeof(double))

    cpdef void compute_pointwise_physics(self) noexcept nogil:
        self.assign_physics()

    cpdef void compute_cellwise_geometry(self,) noexcept nogil:
        self.compute_cell_areas()
    cpdef void compute_cellwise_physics(self,) noexcept nogil:
        pass

    cpdef void compute_cell_areas(self,) noexcept nogil:
        if self.mesh_type==MeshType.TRI:
            __assign_position_difference_vectors(self.nf, self._xyz, self._f_v_i, self.db_area_vector_pairs)
            __assign_triangle_area(self.nf, self.db_area_vector_pairs, self._cell_areas)
        elif self.mesh_type==MeshType.QUAD:
            pass
            # self.cell_areas = compute_cell_area_quad(self.xyz, self.f_v_i)

    cpdef void inclination(self, double incl) noexcept nogil:
        cdef double u = deg2rad*incl
        cdef double c = cos(u)
        cdef double s = sin(u)
        # For BLAS 
        self.rotmat[0] = c
        self.rotmat[1] = 0
        self.rotmat[2] = s
        self.rotmat[3] = 0
        self.rotmat[4] = 1
        self.rotmat[5] = 0
        self.rotmat[6] = -s
        self.rotmat[7] = 0
        self.rotmat[8] = c

    cpdef void inclination_by_matrix(self, double cosine, double sine) noexcept nogil:
        # For BLAS 
        self.rotmat[0] = cosine
        self.rotmat[1] = 0
        self.rotmat[2] = sine
        self.rotmat[3] = 0
        self.rotmat[4] = 1
        self.rotmat[5] = 0
        self.rotmat[6] = -sine
        self.rotmat[7] = 0
        self.rotmat[8] = cosine
        
    cpdef void rotate_y(self, double angle,) noexcept nogil:
        self.incl = angle
        self.inclination(self.incl)
        self._rotate_backend_private()

    cpdef void rotate_y_cosine(self, double cosine,) noexcept nogil:
        cdef double sine = sqrt(1 - cosine*cosine)
        self.incl = acos(cosine)*180./M_PI
        self.inclination_by_matrix(cosine, sine)
        self._rotate_backend_private()

    cdef void _rotate_backend_private(self) noexcept nogil:
        # These two are separate from vector properties, so treated differently.
        lia3d_mv_rot_A_0i(self._xyz,    self._xyz0,    self.rotmat[0], self.rotmat[2], 1, self.nv)
        lia3d_mv_rot_A_0i(self._normal, self._normal0, self.rotmat[0], self.rotmat[2], 1, self.nv)
        self.VecDens.rotate_ptr(self.rotmat)
        self.VecIntg.rotate_ptr(self.rotmat)

    cpdef void reorder_by_tau(self) noexcept nogil:
        cdef long i
        for i in prange(self.nv, nogil=True):
            self.__b_idx_v[i] = i
        memcpy(
            <void *>self.db_nv, 
            <void *>self.ScalDens.__data[self.get_density_idx(DensityKeys.TIME_LAG)],
            self.nv*sizeof(double)
        )
        __reorder_based_on_dblscalar(
            self.nv, self.nf, 
            self._xyz, self._f_v_i, self._v_f_i, self._v_f_n, 
            self.db_nv, self.__b_idx_v, self.lb_nv, 
        )
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, self.__b_idx_v, self._xyz0)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, self.__b_idx_v, self._normal)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, self.__b_idx_v, self._normal0)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, self.__b_idx_v, self._resp_raw) # Is this necessary?
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, self.__b_idx_v, self._response)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, self.__b_idx_v, self._emis_raw) # Is this necessary?
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, self.__b_idx_v, self._emission)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, self.__b_idx_v, self._rho)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, self.__b_idx_v, self._R)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, self.__b_idx_v, self._RLamp)
        
        for i in range(self.ScalDens.N):
            assign_sorted_scalar_inplace_dbl_ptr(self.nv, self.__b_idx_v, self.ScalDens.__data[i])
        for i in range(self.VecDens.N):
            assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, self.__b_idx_v, self.VecDens.__data_agn_coord[i])
            assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, self.__b_idx_v, self.VecDens.__data_obs_coord[i])
    
    cpdef object reorder_vertices_by(self, 
        double [:] ordering_scalar_arr # nv
    ):  
        cdef double * ordering_scalar = <double *> &ordering_scalar_arr[0]
        cdef long * idx = <long *> malloc(self.nv*sizeof(long))
        cdef long i
        for i in prange(self.nv, nogil=True):
            idx[i] = i
        __reorder_based_on_dblscalar(
            self.nv, self.nf, 
            self._xyz, self._f_v_i, self._v_f_i, self._v_f_n, 
            ordering_scalar, idx, self.lb_nv,
        )
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, idx, self._xyz0)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, idx, self._normal)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, idx, self._normal0)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, idx, self._resp_raw) # Is this necessary?
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, idx, self._response)
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, idx, self._emis_raw) # Is this necessary?
        assign_sorted_vector_inplace_dbl_ptr(self.nv, self.ntfs, idx, self._emission)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, idx, self._rho)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, idx, self._R)
        assign_sorted_scalar_inplace_dbl_ptr(self.nv, idx, self._RLamp)
        for i in range(self.ScalDens.N):
            assign_sorted_scalar_inplace_dbl_ptr(self.nv, idx, self.ScalDens.__data[i])
        for i in range(self.VecDens.N):
            assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, idx, self.VecDens.__data_agn_coord[i])
            assign_sorted_vector_inplace_dbl_ptr(self.nv, 3, idx, self.VecDens.__data_obs_coord[i])
        return numpy_lng_1d(idx, self.nv, True)

    cdef void __trim_mesh_geometry(self, long * out_nv_good, long * out_nf_good, long * out_ninterp) noexcept nogil:
        _c_trim_mesh(
            self.nv, self.nf,
            self._xyz,
            self._f_v_i, self._v_f_i, self._v_f_n, self._cell_areas,
            # buffers, needs to be replaced with self.
            self.mask_v, self.mask_f, self.mask_potential,
            self.lb_nv, self.lb_nf,
            # outputs
            out_nv_good, out_nf_good, out_ninterp,
            self.__b_xyz, self.__b_f_v_i, self.__b_v_f_i, self.__b_v_f_n, self.__b_area,
            self.__b_idx_v, self.__b_idx_f,
            self.__b_nip, self.__b_iip, self.__b_iw, 
        )
    cdef void __trim_mesh_physics(self, long nv_good, long nf_good, long ninterp) noexcept nogil:  
        _c_reorder_scalar_after_trim(
            self.nv, self.__b_tau,  self.ScalDens.__data[self.get_density_idx(DensityKeys.TIME_LAG)],
            nv_good, self.__b_idx_v, ninterp, self.__b_nip, self.__b_iip, self.__b_iw,
            self.db_nv,
        )
        _c_reorder_vector_after_trim(
            self.nv, self.ntfs, self.__b_resp, self._response,
            nv_good, self.__b_idx_v, ninterp, self.__b_nip, self.__b_iip, self.__b_iw,
            self.db_nvntfs,
        )
        _c_reorder_vector_after_trim(
            self.nv, self.ntfs, self.__b_emis, self._emission,
            nv_good, self.__b_idx_v, ninterp, self.__b_nip, self.__b_iip, self.__b_iw,
            self.db_nvntfs,
        )
    cpdef tuple trim_mesh_geometry(self):
        cdef long nv_good, nf_good, ninterp
        with nogil:
            self.get_obscuration_mask() # This is not part of __trim_mesh, but it is required anyways.
            self.__trim_mesh_geometry(
                &nv_good, &nf_good, &ninterp
            )
        return nv_good, nf_good, ninterp
    cpdef void trim_mesh_physics(self, long nv_good, long nf_good, long ninterp):
        with nogil:
            self.__trim_mesh_physics(
                nv_good, nf_good, ninterp
            )
    cdef void __trim_mesh(self, long * out_nv_good, long * out_nf_good) noexcept nogil:  
        cdef long ninterp
        self.__trim_mesh_geometry(
            out_nv_good, out_nf_good, &ninterp
        )
        self.__trim_mesh_physics(
            out_nv_good[0], out_nf_good[0], ninterp
        )
    
    cpdef void obscuration_potential(self) noexcept nogil:  
        cdef long i
        cdef double * xyz
        cdef double xc, yc
        cdef double * out = self.ScalDens.__data[self.get_density_idx(DensityKeys.OBSC_POTENTIAL)]
        cdef double s_f = sqrt(1-self.c_f*self.c_f)
        cdef double cy1 = 1./(self.R_out*s_f)
        cdef double cx1 = cy1/self.rotmat[0]
        cdef double cx0 = - (self.c_f*self.rotmat[2])/(s_f*self.rotmat[0])
        for i in prange(self.nv, nogil=True): # USE BLAS - LEVEL 1?
            xyz = self._xyz + 3*i
            xc = cx0 + cx1*xyz[0]
            yc =       cy1*xyz[1]
            out[i] = sqrt(xc*xc + yc*yc) - 1 # < 0 inside the bowl: Outside bowl means self-obscured.
            # out[i] = xc*xc + yc*yc - 1 # < 0 This is less desirable than above since it is not exactly equi-distant. 
    
    cdef void get_obscuration_mask(self) noexcept nogil:
        # self.db_nv: normal_z
        lia_v_comp(self.db_nv, self._normal, 2, 3, self.nv) # Copy instead of inner product with unit_z
        make_obscuration_mask(
            self.nv, self.nf, self.nvpf, self._f_v_i, self._v_f_i, self._v_f_n,
            self.ScalDens.__data[self.get_density_idx(DensityKeys.OBSC_POTENTIAL)], self.db_nv,
            self.mask_v, self.mask_f, self.mask_potential,
        )

    cdef void interpolate_and_reorder_scalar_density(self,
        double * outarr, str s_str, 
        long n_good, long ninterp,
        long * idx_order,
    ) noexcept nogil:
        cdef long idx
        with gil:
            idx = self.ScalDens.entries.get_index(s_str)
        _c_reorder_scalar_after_trim(
            self.nv, outarr, self.ScalDens.__data[idx], 
            n_good, idx_order, ninterp, self.__b_nip, self.__b_iip, self.__b_iw,
            self.db_nv,
        )

    cpdef void assign_time_lag(self) noexcept nogil:
        cdef double * tau = self.ScalDens.__data[self.get_density_idx(DensityKeys.TIME_LAG)]
        lia_vv_sub(
            tau,                    # tau_rest
            # self._R, self._xyz + 2, # = R - xyz[:,2] = R - z
            self._RLamp, self._xyz+2, # = R(lamp-vert) - xyz[:,2] = R(lamp-vert) - z
            self.nv,                # in total, nv elements
            stride_x=1, stride_y=3, # R is scalar, xyz is 3-dim vector
        )
        lia_sv_scalar_inplace(      # tau_obs
            exp(+self.log1pz), tau, # = (1+z) * tau_rest
            self.nv,                # in total, nv elements
            stride_x=1              # tau is scalar
        )

    cpdef void assign_response_obs(self) noexcept nogil:
        # This applies the viewing angle factor.
        cdef size_t i
        if self.c_f == 0 or self._Rpacket.pars.mod_refl == ReflectionModel.REF_ISOTROPIC:
            memcpy(<void *>self._response, <void *>self._resp_raw, self.nv*self.ntfs*sizeof(double))
            memcpy(<void *>self._emission, <void *>self._emis_raw, self.nv*self.ntfs*sizeof(double))
        else:
            if self._Rpacket.pars.mod_refl == ReflectionModel.REF_LAMBERTIAN:
                for i in range(self.nv):
                    self.db_nv[i] = self._normal[3*i + 2]
            elif self._Rpacket.pars.mod_refl == ReflectionModel.REF_LAMBERTIAN_SQRT:
                for i in range(self.nv):
                    self.db_nv[i] = sqrt(self._normal[3*i + 2])
            for i in range(self.ntfs):
                lia_vv_mul(
                    self._response + i,
                    self._resp_raw + i,
                    self.db_nv,
                    self.nv,             # in total, nv elements
                    stride_x=self.ntfs,  # stride for input and output are ntfs.
                    stride_y=1,
                    stride_out=self.ntfs,
                )
                lia_vv_mul(
                    self._emission + i,
                    self._emis_raw + i,
                    self.db_nv,
                    self.nv,             # in total, nv elements
                    stride_x=self.ntfs,  # stride for input and output are ntfs.
                    stride_y=1,
                    stride_out=self.ntfs,
                )
    cpdef object compute_temperatures(self):
        cdef double * temp_at_vert = <double *> malloc(self.nv*sizeof(double))
        with nogil:
            T_all(self.nv, temp_at_vert, self.resp_data, self._Rpacket,)
        return numpy_dbl_1d(temp_at_vert, self.nv, True)

    cdef void __get_tf(self, # API for MCMC
        long n_resp_tau, 
        # Output
        double * resps_1d_pdf, # n_resp_tau*ntfs
        # Time variables
        double * tau_resps,    # n_resp_tau
        double * d_tau_bins,   # n_resp_tau
        # Buffers
        double * resps_1d_cdf, # n_resp_tau*ntfs
    ) noexcept nogil:
        cdef long nv_good, nf_good
        self.get_obscuration_mask()
        self.__trim_mesh(
            &nv_good, &nf_good,
        )
        compute_tf_pdf(
            n_resp_tau, self.ntfs, resps_1d_pdf, resps_1d_cdf, 
            tau_resps, d_tau_bins,
            nv_good, nf_good, self.nvpf, 
            self.__b_xyz, self.__b_f_v_i, self.__b_v_f_i, self.__b_v_f_n, 
            self.__b_area, self.__b_resp, self.__b_emis, self.__b_tau, 
            self.__b_precomp, self.__b_average, self.__b_cumulsum, self.__b_mult, self.__b_tau_f,
            self.__b_idx_f, self.lb_nf, self.db_nf, self.db_nfntfs, self.db_ntfs,
            self.mask_v, self.mask_f, self.mask_potential,
        )

    cdef void expose_packet(self, packet_Mesh1D * packet) noexcept nogil:
        # Should only be run once in theory.
        packet.nvpf           = self.nvpf
        packet.ntfs           = self.ntfs
        packet.__b_xyz        = self.__b_xyz
        packet.__b_f_v_i      = self.__b_f_v_i
        packet.__b_v_f_i      = self.__b_v_f_i
        packet.__b_v_f_n      = self.__b_v_f_n
        packet.__b_area       = self.__b_area
        packet.__b_resp       = self.__b_resp
        packet.__b_emis       = self.__b_emis
        packet.__b_tau        = self.__b_tau
        packet.__b_precomp    = self.__b_precomp
        packet.__b_average    = self.__b_average
        packet.__b_cumulsum   = self.__b_cumulsum
        packet.__b_mult       = self.__b_mult
        packet.__b_tau_f      = self.__b_tau_f
        packet.__b_idx_f      = self.__b_idx_f
        packet.lb_nf          = self.lb_nf
        packet.db_nf          = self.db_nf
        packet.db_nfntfs      = self.db_nfntfs
        packet.db_ntfs        = self.db_ntfs
        packet.mask_v         = self.mask_v
        packet.mask_f         = self.mask_f
        packet.mask_potential = self.mask_potential

    cdef void compute_tf(self, # Frontend for stand-alone computation
        long n_resp_tau, double * tau_resps, 
        double incl,
        double * resps_1d_pdf
    ) noexcept nogil:
        # Buffer variables owned by class that owns "tau_bin_edges" and other likelihood functions
        cdef double * resps_1d_cdf = <double *> malloc(n_resp_tau * self.ntfs * sizeof(double))
        ## Persistent variable over all likelihood computations
        cdef double * d_tau_bins   = <double *> malloc(n_resp_tau * sizeof(double))  # Precomp when fitting
        # finite_difference_f(n_resp_tau+1, tau_bin_edges, d_tau_bins) # Precomp when fitting
        finite_difference_b(n_resp_tau, tau_resps, d_tau_bins) # Precomp when fitting
        self.rotate_y(incl)
        self.assign_time_lag()
        self.assign_response_obs()
        self.reorder_by_tau()
        self.obscuration_potential()
        self.__get_tf(n_resp_tau, resps_1d_pdf, tau_resps, d_tau_bins, resps_1d_cdf)
        free(resps_1d_cdf)
        free(d_tau_bins) # Precomp when fitting

    cpdef object get_tf(self, 
        double [:] tau_view, double incl, 
    ):  
        cdef long n_resp_tau = tau_view.shape[0]
        cdef double * resps_1d_pdf = <double *> malloc(n_resp_tau*self.ntfs*sizeof(double))
        cdef double * tau_resps = <double *> &tau_view[0]
        self.compute_tf(n_resp_tau, tau_resps, incl, resps_1d_pdf)
        # return numpy_dbl_1d(resps_1d_pdf, n_resp_tau, True)
        # return numpy_dbl_2d(resps_1d_pdf, n_resp_tau, self.ntfs, True)
        return numpy_dbl_2d(resps_1d_pdf, self.ntfs, n_resp_tau, True)

    cdef double _compute_radial_penalty(self) noexcept nogil:
        # Compute Jeffrey Prior based on Fisher Information of increased TF.
        cdef long i
        cdef double penalty, dA
        dA = slim_free_marginal_area(self.R_out, self.rotmat[0], self.c_f, self.curv)
        
        # cdef long i, j, I
        # cdef double rho, r, z, azm
        # r = self._rad1d[self.nrad]
        # z = self._z1d[self.nrad]
        # rho = sqrt(r*r - z*z)
        # self._normal
        # self.__b_mult # normalizing factors (1/y) for each
        # self.__b_cumulsum # cdf of tf (x/y) for each.
        return dA

    cpdef double compute_radial_penalty(self):
        cdef double penalty
        with nogil:
            penalty = self._compute_radial_penalty()
        return penalty
        


    cpdef void set_vector_density(self, str name, double [:, :] data) noexcept nogil:
        self.VecDens.set_in_obs_coord(name, data)
    cpdef void set_vector_integ(self, str name, double [:, :] data) noexcept nogil:
        self.VecIntg.set_in_obs_coord(name, data)
    cpdef void set_scalar_density(self, str name, double [:] data) noexcept nogil:
        self.ScalDens.set(name, data)
    cpdef void set_scalar_integ(self, str name, double [:] data) noexcept nogil:
        self.ScalIntg.set(name, data)
    
    cpdef object get_vector_density(self, str name):
        return self.VecDens.get_obs(name)
    cpdef object get_vector_integ(self, str name):
        return self.VecIntg.get_obs(name)
    cpdef object get_scalar_density(self, str name):
        return self.ScalDens.get(name)
    cpdef object get_scalar_integ(self, str name):
        return self.ScalIntg.get(name)
    
    # Intrinsic coordinate vectors
    cpdef void set_vector_density_intr(self, str name, double [:, :] data) noexcept nogil:
        self.VecDens.set_in_agn_coord(name, data)
    cpdef void set_vector_integ_intr(self, str name, double [:, :] data) noexcept nogil:
        self.VecIntg.set_in_agn_coord(name, data)
    cpdef object get_vector_density_intr(self, str name):
        return self.VecDens.get_agn(name)
    cpdef object get_vector_integ_intr(self, str name):
        return self.VecIntg.get_agn(name)

def constructDiskMesh(
    R_in=None, R_out=None, 
    c_f=None, 
    curv=None, 
    log1pz=0., logLbol=44.09935780, logMass=7., H_lamp=2., pindex=1., LampEff=1., # ln(1 + z)
    nrad=256, nazm=360,
    track_velocity=False,
    mesh_type='tri',
    geom_type='slim',
    resp_type='goad_pow',
    ntfs=1,
    r_logbin=True,
    hexapack=True,
    thermal_distance='radial',
    force_proj=True,
):
    # Defining Parameters
    nrad = int(nrad)
    nazm = int(nazm)
    R_in = float(R_in)
    R_out = float(R_out)
    curv = float(curv)
    ntfs = int(ntfs)
    c_f = float(fabs(c_f))

    mesh = DiskMesh.__new__(DiskMesh,
        nrad, nazm,
        mesh_type=mesh_type,
        geom_type=geom_type,#'thin',#'slim',
        resp_type=resp_type,
        track_velocity=track_velocity,
        store_topology=True,
        thermal_distance=thermal_distance,
        ntfs=ntfs,
        r_logbin=r_logbin,
        hexapack=hexapack,
    )
    mesh.assign_mesh(
        R_in, R_out, c_f, curv,
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, pindex=pindex, LampEff=LampEff,
        force_proj=force_proj,
    )
    return mesh
