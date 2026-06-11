# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from .transformables cimport Scalars, Vectors

from ..physics.response cimport (
    ThermalDistanceScheme,
    par_resp, ResponseModel, ResponsePacket, func_t_resp
)
from ..physics.thermal cimport func_Rsub
from ..utils.types cimport func_t_void_arr_auxarr
from ..geometry.diskgeom cimport func_t_diskgeom, DiskGeometryModel

cdef enum MeshType:
    TRI
    QUAD # NOT IMPLEMENTED

cdef enum DensityKeys:
    TIME_LAG
    # RESP_RAW
    # RESPONSE
    OBSC_POTENTIAL
    # TEMPERATURE
    VELOCITY

cdef struct DensityIndexPair:
    DensityKeys key
    long index

cdef struct packet_Mesh1D:
    # This struct only stores pointer, and should not malloc'd or freed.
    # Mesh Properties
    long nvpf          # constant
    long ntfs          # constant
    double * __b_xyz   # nv, 3
    long   * __b_f_v_i # nf, nvpf
    long   * __b_v_f_i # nv, nfpv
    long   * __b_v_f_n # nv
    double * __b_area  # nf
    double * __b_resp  # nv, ntfs
    double * __b_emis  # nv, ntfs
    double * __b_tau   # nv
    # buffers
    double * __b_precomp  # nf, ntfs
    double * __b_average  # nf, ntfs
    double * __b_cumulsum # nf, ntfs
    double * __b_mult     # ntfs
    double * __b_tau_f    # nf
    long   * __b_idx_f    # nf
    ## Generic Buffers
    long   * lb_nf
    double * db_nf
    double * db_nfntfs
    double * db_ntfs
    # mask buffers
    unsigned char * mask_v
    unsigned char * mask_f
    double * mask_potential
    # # Responses (Don't necessarily need to be here)
    # long ntau          # constant
    # double * resp_pdf  # ntau
    # double * resp_cdf  # ntau, not required for compute_tf.
    # # persistent
    # double * tau_resps  # ntau
    # double * d_tau_bins # ntau
    # long nv_good       # variable: should be passed directly.
    # long nf_good       # variable: should be passed directly.
    

cdef double deg2rad
cdef double[3] unit_z

cdef long find_index_physics(long n, DensityIndexPair *index_to_physics, DensityKeys key) noexcept nogil
# cdef void shallow_copy_packet_Mesh1D(packet_Mesh1D * dst, packet_Mesh1D * src) noexcept nogil
cdef void free_packet_Mesh1D(packet_Mesh1D * packet) noexcept nogil

@cython.final
cdef class DiskMesh():
    # memuse per instance (in bytes):
    # if not topology_is_stored:
    #     368*nrad*nazm + 16*nrad + 238*nazm + extra
    # if topology_is_stored:
    #     381*nrad*nazm + 16*nrad + 245*nazm + extra
    # e.g., for nrad=256, nazm=360 grid, with topology_is_stored=True,
    #     381*256*360 + 16*256 + 245*360 = 35.2MB (33.57MiB) per instance.
    cdef:
        public double incl # Inclination angle, in degrees
        double * rotmat # Rotation matrix from AGN to OBS coordinates
        MeshType mesh_type
        public bint topology_is_stored
        public long nrad
        public long nazm
        public long nv            # Total number of vertices (points)
        public long nf            # Total number of faces (cells)
        public unsigned char nvpf # Number of vertices per face
        public unsigned char nfpv # Maximum number of faces per vertex
        public double curv
        public double R_in
        public double R_out
        public double rho_in
        public double rho_out
        public double c_f
        
        DiskGeometryModel geom_mod
        func_t_diskgeom geom_func

        # Physical properties
        public bint track_velocity
        # public bint T_from_planar
        public double logLbol
        public double logMass
        public double log1pz    # log(1+z)
        public double pindex
        public double H_lamp
        public double Rg
        double H_lamp_ltday
        ThermalDistanceScheme thermal_dist
        public double LampEff

        func_Rsub fRsub

        ResponseModel resp_mod
        func_t_resp resp_func
     
        # Mesh Connectivity Naming Convention:
        # a_b_c
        # a: From where, i.e., the *index* of this array points to a.
        # b: To where,   i.e., the *value* of this array points to b.
        # c: 
        #    If i, the value of this array contains the index to b. 
        #    If n, the value of this array contains the number of valid elements of a_b_i.
        long * _f_v_i # Indices of vertices forming each face, in counter-clockwise order.
        long * _v_f_i # Indices of adjacent faces at each vertex, in no particular order, 
        long * _v_f_n # Maximum number of adjacent faces at given vertex. <= nfpv.
        # v_f_i[nfpv*i + j] only valid until j<v_f_n[i].

        # Geometric Parameters
        double * _xyz0       # Cartesian coordinates of each vertex in AGN coordinates
        double * _normal0    # Normal vectors at each vertex in AGN coordinates
        double * _xyz        # Cartesian coordinates of each vertex in OBS coordinates
        double * _normal     # Normal vectors at each vertex in OBS coordinates
        double * _rho        # Distance of each vertex from the axis of symmetry. Always in AGN coordinates.
        double * _R          # Distance of each vertex from the origin. Coordinate invariant.
        double * _RLamp      # Distance to the lamp post. Coordinate invariant.
        double * _cell_areas # Area of each face

        # Numpy Interface for Mesh
        public object f_v_i
        public object v_f_i
        public object v_f_n
        public object xyz
        public object xyz0
        public object normal
        public object normal0
        public object rho
        public object R
        public object cell_areas

        # Vectors and Scalars associated with the mesh. 
        # Vectors transform between coordinates, while Scalar is coordinate invariant.
        # Dens means density, whose integration is usually not conserved during affine transformations, 
        #   and it is defined at vertices. (e.g., temperature, velocity, ...)
        # Intg means integrated, whose integration should be conserved during affine transformations,
        #   and it is defined at face centers. (e.g., mass, total received power, ...)
        # (Cannot think about a meaningful reason not to do this)
        Vectors VecDens
        Vectors VecIntg
        Scalars ScalDens
        Scalars ScalIntg

        # Index for accessing physical variable. 
        long n_density_dict
        DensityIndexPair * density_dict

        # Responses; don't need to be public.
        # Pointers for passing to physics functions
        # par_resp * p_resp      # Response computation parameters
        void ** resp_data      # Solely for storing pointers.
        ResponsePacket * _Rpacket  # for storing band data
        ## Following will be used for multiple response terms.
        # long n_resp_packets
        # ResponsePacket ** _Rpacket  # for storing band data
        #
        public bint src_azmsym # If the response source term has azmuthal symmetry or not for each source term.
        public long ntfs       # Better to have this as a separate member to be accesible from GIL operations.
        # nvsrc
        #    = (nrad+1) w/  azimuthal symmetry  (src_azmsym=True).
        #    = (nv    ) w/o azimuthal symmetry (src_azmsym=False).
        double * _resp_raw     # Raw response.   size ntfs*nv, order: ntfs is faster.
        double * _response     # Final response. size ntfs*nv, order: ntfs is faster.
        double * _emis_raw     # Raw   total (time-averaged) emission from the disk/torus. size ntfs*nv
        double * _emission     # Final total (time-averaged) emission from the disk/torus. size ntfs*nv
        public object resp_raw
        public object response
        public object emis_raw
        public object emission
        # public long nsrc       # Number of source terms. Included in case if line and continuum should be computed altogether.
        # long * idx_src         # size ntfs, index,

        # Primitive arrays
        double * _rad1d # (nrad+1)
        double * _azm1d #  nazm
        double * _z1d   # (nrad+1)
        # Only if topology_is_stored is True
        long * _f_v_i0  
        long * _v_f_i0  
        long * _v_f_n0 

        # Masks
        unsigned char * mask_v
        unsigned char * mask_f
        double * mask_potential

        # Buffers
        ## Generic Buffers
        long * lb_nv
        long * lb_nf
        double * db_nv
        double * db_nf
        ## Specific Buffers
        double * db_nv3 # for Vectors
        double * db_nvntfs # for Response
        double * db_nfntfs # for Response
        double * db_ntfs   # for response
        double * db_area_vector_pairs # nf*9 for TRI, nf*6 for QUAD (probably?)
        ## Integration buffers
        double * __b_xyz      # nv, 3
        long   * __b_f_v_i    # nf, nvpf
        long   * __b_v_f_i    # nv, nfpv
        long   * __b_v_f_n    # nv
        double * __b_area     # nf
        long   * __b_idx_v    # nv
        long   * __b_idx_f    # nf
        long   * __b_nip      # nv
        long   * __b_iip      # nv, 3
        double * __b_iw       # nv, 2
        double * __b_tau      # nv
        double * __b_resp     # nv, ntfs
        double * __b_emis     # nv, ntfs
        double * __b_precomp  # nf, ntfs
        double * __b_average  # nf, ntfs
        double * __b_cumulsum # nf, ntfs
        double * __b_tau_f    # nf
        double * __b_mult     # ntfs

    cpdef void thread_mesh(self) noexcept nogil
    cpdef void assign_mesh(self, 
        double R_in, double R_out, double c_f, double curv,
        double log1pz=*, double logLbol=*, double logMass=*, double H_lamp=*, double pindex=*, double LampEff=*,
        bint force_proj=*,
        # bint r_logbin=*, bint hexapack=*, # moved to cinit.
    ) noexcept nogil
    cdef void reset_rotation(self) noexcept nogil
    # cdef void assign_coordinates(self) noexcept nogil
    cdef void assign_band_M(self, double * wave) nogil
    cdef void assign_band_I(self, double * x_rb, double * coef_r, double * coef_e, long n_rb) nogil
    cpdef void assign_band(self, object x, object coef_r=*, object coef_e=*)
    # cpdef void update_mesh(self, 
    #     double R_in, double R_out, double c_f, double curv,
    #     double log1pz=*, double logLbol=*, double logMass=*, double H_lamp=*, double pindex=*,
    # ) noexcept nogil
    # cpdef void update_geometry(self, double R_in, double R_out, double c_f, double curv) noexcept nogil
    # cpdef void update_physics(self, double log1pz=*, double logLbol=*, double logMass=*, double H_lamp=*, double pindex=*,) noexcept nogil
    cpdef void update_mesh(self, 
        double R_in, double R_out, double c_f, double curv, 
        double log1pz=*, double logLbol=*, double logMass=*, double H_lamp=*, double pindex=*, double LampEff=*,
    ) noexcept nogil
    cpdef void update_geometry(self, double R_in, double R_out, double c_f, double curv, double H_lamp) noexcept nogil
    cpdef void update_physics(self, double log1pz=*, double logLbol=*, double logMass=*, double H_lamp=*, double pindex=*, double LampEff=*) noexcept nogil
    cdef void _update_geometry(self) noexcept nogil
    cdef void _update_physics(self) noexcept nogil
    cpdef void assign_dict_indices(self) noexcept
    cpdef long get_density_idx(self, DensityKeys key) noexcept nogil
    # cpdef void compute_distances(self,) noexcept nogil
    # cpdef void compute_normals(self) noexcept nogil
    cdef void assign_velocity(self) noexcept nogil
    cdef void assign_response_int(self) noexcept nogil
    cpdef void assign_physics(self) noexcept nogil
    # cpdef void compute_pointwise_geometry(self) noexcept nogil
    cpdef void compute_pointwise_physics(self) noexcept nogil
    cpdef void compute_cellwise_geometry(self,) noexcept nogil
    cpdef void compute_cellwise_physics(self,) noexcept nogil
    cpdef void compute_cell_areas(self,) noexcept nogil
    cpdef void inclination(self, double incl) noexcept nogil
    cpdef void inclination_by_matrix(self, double cosine, double sine) noexcept nogil
    cpdef void rotate_y(self, double angle,) noexcept nogil
    cpdef void rotate_y_cosine(self, double cosine,) noexcept nogil
    cdef void _rotate_backend_private(self) noexcept nogil
    cpdef void reorder_by_tau(self) noexcept nogil
    cpdef object reorder_vertices_by(self, double [:] ordering_scalar_arr)
    cdef void __trim_mesh(self, long * out_nv_good, long * out_nf_good) noexcept nogil
    cdef void __trim_mesh_geometry(self, long * out_nv_good, long * out_nf_good, long * out_ninterp) noexcept nogil
    cdef void __trim_mesh_physics(self, long nv_good, long nf_good, long ninterp) noexcept nogil
    cpdef tuple trim_mesh_geometry(self)
    cpdef void trim_mesh_physics(self, long nv_good, long nf_good, long ninterp)
    cpdef void obscuration_potential(self) noexcept nogil
    cdef void get_obscuration_mask(self) noexcept nogil
    cdef void interpolate_and_reorder_scalar_density(self, 
        double * outarr, str s_str, long n_good, long ninterp, long * idx_order,) noexcept nogil
    cpdef void assign_time_lag(self) noexcept nogil
    cpdef void assign_response_obs(self) noexcept nogil
    cpdef object compute_temperatures(self)
    cdef void __get_tf(self, # API for MCMC
        long n_resp_tau, 
        # Output
        double * resps_1d_pdf,  # n_resp_tau
        # Time variables
        double * tau_bin_edges, # n_resp_tau + 1
        double * d_tau_bins,    # n_resp_tau + 1
        # Buffers
        double * resps_1d_cdf,  # n_resp_tau + 1
    ) noexcept nogil
    cdef void expose_packet(self, packet_Mesh1D * packet) noexcept nogil
    cdef void compute_tf(self, # Frontend for stand-alone computation
        long n_resp_tau, double * tau_bin_edges, 
        double incl,
        double * resps_1d_pdf
    ) noexcept nogil
    cpdef object get_tf(self, double [:] tau_bin_edges_view, double incl)

    cdef double _compute_radial_penalty(self) noexcept nogil
    cpdef double compute_radial_penalty(self)

    cpdef void set_vector_density(self, str name, double [:, :] data) noexcept nogil
    cpdef void set_vector_integ(self, str name, double [:, :] data) noexcept nogil
    cpdef void set_scalar_density(self, str name, double [:] data) noexcept nogil
    cpdef void set_scalar_integ(self, str name, double [:] data) noexcept nogil
    cpdef object get_vector_density(self, str name)
    cpdef object get_vector_integ(self, str name)
    cpdef object get_scalar_density(self, str name)
    cpdef object get_scalar_integ(self, str name)
    
    # Intrinsic coordinate vectors
    cpdef void set_vector_density_intr(self, str name, double [:, :] data) noexcept nogil
    cpdef void set_vector_integ_intr(self, str name, double [:, :] data) noexcept nogil
    cpdef object get_vector_density_intr(self, str name)
    cpdef object get_vector_integ_intr(self, str name)
