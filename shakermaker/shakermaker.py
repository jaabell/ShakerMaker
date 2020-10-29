import copy
import numpy as np
import logging
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker import core 


class ShakerMaker:

    def __init__(self, crust, source, receivers):
        assert isinstance(crust, CrustModel), \
            "crust must be an instance of the shakermaker.CrustModel class"
        assert isinstance(source, FaultSource), \
            "source must be an instance of the shakermaker.FaultSource class"
        assert isinstance(receivers, StationList), \
            "receivers must be an instance of the shakermaker.StationList class"

        self._crust = crust
        self._source = source
        self._receivers = receivers

        self._mpi_rank = None
        self._mpi_nprocs = None
        self._logger = logging.getLogger(__name__)

    def run(self, dt=0.05, nfft=4096, tb=1000, smth=1, sigma=2, taper=0.9, wc1=1, wc2=2, pmin=0, pmax=1, dk=0.3,
            nx=1, kc=15.0, writer=None):
        self._logger.info('ShakerMaker.run - starting\n\tNumber of sources: {}\n\tNumber of receivers: {}\n'
                          '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
                          .format(self._source.nsources, self._receivers.nstations,
                                  self._source.nsources*self._receivers.nstations, dt, nfft))
        if writer:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers)
            writer.write_metadata(self._receivers.metadata)

        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):
                aux_crust = copy.deepcopy(self._crust)

                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                tdata, z, e, n, t0 = self._call_core(dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                     taper, aux_crust, psource, station)

                t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                psource.stf.dt = dt/10
                z_stf = psource.stf.convolve(z)
                e_stf = psource.stf.convolve(e)
                n_stf = psource.stf.convolve(n)

                station.add_to_response(z_stf, e_stf, n_stf, t)

            self._logger.debug('ShakerMaker.run - finished station {}'.format(i_station))

            if writer:
                writer.write_station(station, i_station)

        if writer:
            writer.close()

    def write(self, writer):
        writer.write(self._receivers)

    def enable_mpi(self, rank, nprocs):
        self._mpi_rank = rank
        self._mpi_nprocs = nprocs

    def mpi_is_master_process(self):
        return self.mpi_rank == 0

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        return self._mpi_nprocs

    def _call_core(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, crust, psource, station):
        mb = crust.nlayers
        print(f"psource = {psource}")
        print(f"psource.x = {psource.x}")
        print(f"station = {station}")
        print(f"station.x = {station.x}")

        src = crust.get_layer(psource.x[2]) + 1   # fortran starts in 1, not 0
        rcv = crust.get_layer(station.x[2]) + 1   # fortran starts in 1, not 0
        
        stype = 2  # Source type double-couple, compute up and down going wave
        updn = 0
        d = crust.d
        a = crust.a
        b = crust.b
        rho = crust.rho
        qa = crust.qa
        qb = crust.qb

        pf = psource.angles[0]
        df = psource.angles[1]
        lf = psource.angles[2]
        sx = psource.x[0]
        sy = psource.x[1]
        rx = station.x[0]
        ry = station.x[1]
        x = np.sqrt((sx-rx)**2 + (sy - ry)**2)

        self._logger.debug('ShakerMaker._call_core - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
                           '\tstyoe: {}\n\tupdn: {}\n\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                           '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n\tsmth: {}\n\twc1: {}\n\twc2: {}\n'
                           '\tpmin: {}\n\tpmax: {}\n\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n\tpf: {}\n\tdf: {}\n'
                           '\tlf: {}\n\tsx: {}\n\tsy: {}\n\trx: {}\n\try: {}\n\t'
                           .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1,
                                   wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))

        # Execute the core subgreen fortran routing
        tdata, z, e, n, t0 = core.subgreen(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma,
                                           smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry)

        self._logger.debug('ShakerMaker._call_core - core.subgreen returned: z_size'.format(len(z)))

        return tdata, z, e, n, t0
