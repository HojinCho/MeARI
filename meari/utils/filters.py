import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import typing
import h5py

from . import PPI3_param
from .. import __version__

class ResponseTabulator:
    def __init__(
        self,
        filename: str, 
        filter_name: typing.Optional[str]=None,
        backend: typing.Literal['numpy', 'pandas']='numpy',
        backend_args: typing.Optional[typing.Union[tuple, list]]=None,
        backend_kwargs: typing.Optional[dict]=None,
    ):
        if backend=='pandas':
            if pd is None:
                raise ImportError('pandas is not installed. Please install pandas to use this backend.')
            if backend_kwargs is None:
                backend_kwargs = {}
            if backend_args is None:
                backend_args = []
            data = pd.read_csv(filename, *backend_args, **backend_kwargs)
            wave, tran = data[data.columns[0]].values, data[data.columns[1]].values
        if backend=='numpy':
            if backend_kwargs is None:
                kwargs = {}
            else:
                kwargs = dict(**backend_kwargs)
            kwargs['unpack'] = True
            if backend_args is None:
                backend_args = []
            data = np.genfromtxt(filename, *backend_args, **kwargs)
            wave, tran = data[0], data[1]
        self.wave = wave.copy()
        self.tran = tran.copy()
        self.table = {
            'T': None,
            'flux': None,
            'resp': None,
        }
        self.ppi3 = {
            'N_rb':      0,
            'T_rb':   None,
            'F_coef': None,
            'R_coef': None,
            'T_log':  None,
            'F_log':  None,
            'R_log':  None,
            'T_min':  None,
            'T_max':  None,
        }
        if filter_name is None:
            self.name = filename.split('/')[-1].split('.')[0]
        else:
            self.name = filter_name
        self.sourcefile = filename
        return
    
    def tabulate_nodes(
        self,
        n: int=68, Tmin: float=1e2, Tmax: float=1e6,
        spacing: typing.Literal['linear', 'log']='log',
        integ_scheme: typing.Literal['simpson', 'trapezoidal', 'quad', 'riemann_l', 'riemann_r']='simpson', 
        resample_scheme: typing.Optional[typing.Literal['linear', 'cubic']]='cubic', 
        resample_zeropad: int=3,
        resample_n: typing.Optional[int]=100000,
        epsrel: float=1e-4,
    ):
        # Import and Function declarations are placed here to avoid unintended importing
        from scipy.integrate import trapezoid, simpson, quad
        from scipy.interpolate import make_interp_spline
        def planck_resp(wave, T):
            # C1_BBresponse = 4511216.452581668706676772402
            C1_BBresponse = 3.0266524669317244421289896765449e+29
            C2_BBresponse = 14387.76877503933790167601328
            invBoltz = np.exp(C2_BBresponse/(wave*T))
            return C1_BBresponse*(1./(wave*T)**5)*((invBoltz)/(invBoltz-1.)**2)
        def planck_flux(wave, T):
            # Photon flux
            cm_to_ltday = 3.86069554627490798E-16
            um_to_cm = 1e-4
            c0 = 29979245800.0
            hckB = 1.4387768775039337901676013279982
            # cm = um * (d cm/d um)
            wv = um_to_cm*wave
            # F_cm d cm = F_um d um -> F_um = F_cm * (d cm/d um)
            return um_to_cm*( # to convert back to /um scale
                (2*c0/(wv)**4) # numerator, divided by Ephoton
                /(np.exp(hckB/(wv*T)) - 1) # denominator
            )/cm_to_ltday**2
        # Assigning output memory
        self.table['flux'] = np.empty(n, dtype=np.float64)
        self.table['resp'] = np.empty(n, dtype=np.float64)
        if spacing=='linear':
            self.table['T'] = np.linspace(Tmin, Tmax, n)
        elif spacing=='log':
            self.table['T'] = np.geomspace(Tmin, Tmax, n)
        # Defining integrand
        if resample_scheme is not None:
            if resample_zeropad<0:
                raise ValueError('resample_zeropad must be greater than or equal to 0')
            idx = np.nonzero(self.tran)[0]
            id_min, id_max = max(np.min(idx)-resample_zeropad, 0), min(np.max(idx)+resample_zeropad, self.tran.size-1)
            w, t = self.wave[id_min:id_max+1], self.tran[id_min:id_max+1]
            x = np.linspace(w[0], w[-1], resample_n)
            if resample_scheme=='cubic':
                y = make_interp_spline(w, t, k=3, bc_type='clamped')(x)
            else:
                y = np.interp(x, w, t)
        else:
            x, y = self.wave, self.tran
        # Integration scheme
        if integ_scheme=='quad':
            integrator = (
                lambda T, func: quad(
                    (lambda wv: make_interp_spline(x, y, k=3, bc_type='clamped')(wv)*func(wv, T)), x[0], x[-1], 
                    epsabs=0, epsrel=epsrel, # Absolute error is hardly useful since the approximate value is unpredictable.
                )[0]
            )
        elif integ_scheme=='simpson':
            integrator = (
                lambda T, func: simpson(y*func(x, T), x=x)
            )
        elif integ_scheme=='trapezoidal':
            integrator = (
                lambda T, func: trapezoid(y*func(x, T), x=x)
            )
        elif 'riemann' in integ_scheme:
            dx = x[1:]-x[:-1]
            if integ_scheme[-1]=='l':
                integrator = (
                    lambda T, func: np.sum(y[:-1]*func(x[:-1], T)*dx)
                )
            elif integ_scheme[-1]=='r':
                integrator = (
                    lambda T, func: np.sum(y[1:]*func(x[1:], T)*dx)
                )
        for i, T in enumerate(self.table['T']):
            self.table['flux'][i] = integrator(T, planck_flux)
            self.table['resp'][i] = integrator(T, planck_resp)
        return
    
    def compute_ppi3(
        self, recycle_table: bool=False, # Highly discouraged
        n_node: int=64, Tmin: float=1e2, Tmax: float=1e6,
        T_log: bool=True, F_log: bool=True, R_log: bool=True,
        integ_scheme: typing.Literal['simpson', 'trapezoidal', 'quad', 'riemann_l', 'riemann_r']='simpson', 
        resample_scheme: typing.Optional[typing.Literal['linear', 'cubic']]='cubic', 
        resample_zeropad: int=3,
        resample_n: typing.Optional[int]=100000,
        epsrel: float=1e-4,
    ):
        if not recycle_table or self.table['resp'] is None:
            if T_log:
                xsp = 'log'
            else:
                xsp = 'linear'
            self.tabulate_nodes(
                n=n_node+4, Tmin=Tmin, Tmax=Tmax,
                spacing=xsp, integ_scheme=integ_scheme,
                resample_scheme=resample_scheme, 
                resample_zeropad=resample_zeropad,
                resample_n=resample_n, epsrel=epsrel,
            )
        if T_log:
            x = np.log(self.table['T'])
        else:
            x = self.table['T']
        if R_log:
            y_R = np.log(self.table['resp'])
        else:
            y_R = self.table['resp']
        if F_log:
            y_F = np.log(self.table['flux'])
        else:
            y_F = self.table['flux']
        ppi3_R = PPI3_param(x, y_R)
        ppi3_F = PPI3_param(x, y_F)
        self.ppi3['T_rb'] = ppi3_R[0]
        self.ppi3['R_coef'] = ppi3_R[1]
        self.ppi3['F_coef'] = ppi3_F[1]
        self.ppi3['N_rb'] = int(ppi3_R[0].size)
        self.ppi3['T_min'] = float(np.min(x))
        self.ppi3['T_max'] = float(np.max(x))
        self.ppi3['T_log'] = T_log
        self.ppi3['R_log'] = R_log
        self.ppi3['F_log'] = F_log
        return

    def __call__(self, toget: typing.Literal['node', 'ppi3']='node'):
        if toget=='node':
            if self.table['resp'] is None:
                self.tabulate_nodes()
            return self.table['T'], self.table['flux'], self.table['resp']
        else:
            if self.ppi3['R_coef'] is None:
                self.compute_ppi3()
            return self.ppi3['T_rb'], self.ppi3['F_coef'], self.ppi3['R_coef']

    def write_nodes(
        self,
        outfile: typing.Optional[str]=None,
        directory: str='responses/',
        overwrite: bool=False,
        timestamp: bool=True,
        time_in_utc: bool=False,
    ):
        if self.table['resp'] is None:
            self.tabulate_nodes()
        import datetime
        import pathlib
        d = pathlib.Path(directory)
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        elif not d.is_dir():
            raise ValueError(f'{directory} is not a directory')
        if outfile is None:
            outfile = pathlib.PurePath(f'{self.name}.dat')
        else:
            outfile = pathlib.PurePath(f"{pathlib.PurePath(outfile).name.stem}.dat")
        fn = pathlib.Path(d, outfile)
        if fn.exists() and not overwrite:
            raise FileExistsError(f'{fn} already exists. Use overwrite=True to overwrite.')
        output_dat = np.log(self.table['T']), np.log(self.table['resp'])
        with open(fn, 'w') as fp:
            fp.write("# MeARI Response Tabulation\n")
            if timestamp:    
                if time_in_utc:
                    fp.write(f"# Written on {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
                else:
                    fp.write(f"# Written on {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n")
            fp.write(f"# Filter: {self.name}\n")
            fp.write(f"# N_pts : {len(output_dat[0]):d}\n")
            fp.write(f"# Source: {self.sourcefile}\n")
            fp.write("# \n")
            fp.write("# log T[K]\tlog dN_{phot,resp}/dN_{phot,driv} [cgs]\n")
            for i in range(len(output_dat[0])):
                fp.write(f"{output_dat[0][i]:.17e}\t{output_dat[1][i]:.17e}\n")
            fp.close()
        return
    
    def write_ppi3(
        self,
        outfile: typing.Optional[str]=None,
        directory: str='ppi3/',
        overwrite: bool=False,
        timestamp: bool=True,
        time_in_utc: bool=False,
    ):
        if self.ppi3['R_coef'] is None:
            self.compute_ppi3()
        import datetime
        import pathlib
        d = pathlib.Path(directory)
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        elif not d.is_dir():
            raise ValueError(f'{directory} is not a directory')
        if outfile is None:
            outfile = pathlib.PurePath(f'{self.name}.hdf5')
        else:
            outfile = pathlib.PurePath(f"{pathlib.PurePath(outfile).name.stem}.hdf5")
        fn = pathlib.Path(d, outfile)
        if fn.exists() and not overwrite:
            raise FileExistsError(f'{fn} already exists. Use overwrite=True to overwrite.')
        
        with h5py.File(fn, "w") as fp:
            # See https://www.uetke.com/blog/python/how-to-use-hdf5-files-in-python/
            # For a general guide to hdf5.
            grp = fp.create_group("PPI3")
            grp.attrs['description'] = "MeARI Interpolation Table for Blackbody Response Function"
            grp.attrs['MeARI version'] = __version__
            if timestamp:
                if time_in_utc:
                    grp.attrs['Timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                else:
                    grp.attrs['Timestamp'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
            grp.attrs['filtername'] = self.name
            grp.attrs['sourcefile'] = self.sourcefile
            grp.attrs['N_rb' ] = self.ppi3['N_rb' ]
            grp.attrs['T_log'] = self.ppi3['T_log']
            grp.attrs['F_log'] = self.ppi3['R_log']
            grp.attrs['R_log'] = self.ppi3['R_log']
            grp.attrs['T_min'] = self.ppi3['T_min']
            grp.attrs['T_max'] = self.ppi3['T_max']
            grp.create_dataset("T_rb", data=self.ppi3['T_rb'])
            grp.create_dataset("coef", data=self.ppi3['R_coef'])
            grp.create_dataset("coef_flux", data=self.ppi3['F_coef'])
            fp.close()
        return