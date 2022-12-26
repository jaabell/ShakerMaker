import copy
import numpy as np
import logging
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker import core 
import imp

try:
    imp.find_module('mpi4py')
    found_mpi4py = True
except ImportError:
    found_mpi4py = False

if found_mpi4py:
    # print "Found MPI"
    from mpi4py import MPI
    use_mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
else:
    # print "Not-Found MPI"
    rank = 0
    nprocs = 1
    use_mpi = False


class ShakerMaker:
    """This is the main class in ShakerMaker, used to define a model, link components, 
    set simulation  parameters and execute it. 

    :param crust: Crustal model used by the simulation. 
    :type crust: :class:`CrustModel`
    :param source: Source model(s). 
    :type source: :class:`FaultSource`
    :param receivers: Receiver station(s). 


    """
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

        # self._mpi_rank = None
        # self._mpi_nprocs = None        
        self._mpi_rank = rank
        self._mpi_nprocs = nprocs
        self._logger = logging.getLogger(__name__)

    def run(self, 
        dt=0.05, 
        nfft=4096, 
        tb=1000, 
        smth=1, 
        sigma=2, 
        taper=0.9, 
        wc1=1, 
        wc2=2, 
        pmin=0, 
        pmax=1, 
        dk=0.3,
        nx=1, 
        kc=15.0, 
        writer=None,
        verbose=False,
        debugMPI=False):
        """Run the simulation. 
        
        Arguments:
        :param sigma: Its role is to damp the trace (at rate of exp(-sigma*t)) to reduce the wrap-arround.
        :type sigma: double
        :param nfft: Number of time-points to use in fft
        :type nfft: integer
        :param dt: Simulation time-step
        :type dt: double
        :param tb: Num. of samples before the first arrival.
        :type tb: integer
        :param taper: For low-pass filter, 0-1. 
        :type taper: double
        :param smth: Densify the output samples by a factor of smth
        :type smth: double
        :param wc1: (George.. please provide one-line description!)
        :type wc1: double
        :param wc2: (George.. please provide one-line description!)
        :type wc2: double
        :param pmin: Max. phase velocity, in 1/vs, 0 the best.
        :type pmin: double
        :param pmax: Min. phase velocity, in 1/vs.
        :type pmax: double
        :param dk: Sample interval in wavenumber, in Pi/x, 0.1-0.4.
        :type dk: double
        :param nx: Number of distance ranges to compute.
        :type nx: integer
        :param kc: It's kmax, equal to 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at w=0, we require kmax > 10 to make sure we have have summed enough.
        :type kc: double
        :param writer: Use this writer class to store outputs
        :type writer: StationListWriter
        

        """

        if debugMPI:
        	printMPI = lambda args : print(*args)
        else:
        	printMPI = lambda args : None

        self._logger.info('ShakerMaker.run - starting\n\tNumber of sources: {}\n\tNumber of receivers: {}\n'
                          '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
                          .format(self._source.nsources, self._receivers.nstations,
                                  self._source.nsources*self._receivers.nstations, dt, nfft))
        if rank > 0:
            writer = None

        if writer and rank == 0:
            assert isinstance(writer, StationListWriter), \
                "'writer' must be an instance of the shakermaker.StationListWriter class or None"
            writer.initialize(self._receivers, 2*nfft)
            writer.write_metadata(self._receivers.metadata)
        ipair = 0
        if nprocs == 1 or rank == 0:
            next_pair = rank
            skip_pairs = 1
        else :
            next_pair = rank-1
            skip_pairs = nprocs-1

        npairs = self._receivers.nstations*len(self._source._pslist)

        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):
                aux_crust = copy.deepcopy(self._crust)

                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                if ipair == next_pair:
                    if verbose:
                        print(f"rank={rank} nprocs={nprocs} ipair={ipair} skip_pairs={skip_pairs} npairs={npairs} !!")
                    if nprocs == 1 or (rank > 0 and nprocs > 1):

                        if verbose:
                            print("calling core START")
                        tdata, z, e, n, t0 = self._call_core(dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                             taper, aux_crust, psource, station, verbose)
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        # print(f"********* {psource.tt=} {t0=}")
                        t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                        psource.stf.dt = dt


                        z_stf = psource.stf.convolve(z)
                        e_stf = psource.stf.convolve(e)
                        n_stf = psource.stf.convolve(n)
                        if rank > 0:
                            ant = np.array([nt], dtype=np.int32).copy()
                            printMPI(f"Rank {rank} sending to P0 1")
                            comm.Send(ant, dest=0, tag=2*ipair)
                            data = np.empty((nt,4), dtype=np.float64)
                            printMPI(f"Rank {rank} done sending to P0 1")
                            data[:,0] = z_stf
                            data[:,1] = e_stf
                            data[:,2] = n_stf
                            data[:,3] = t
                            printMPI(f"Rank {rank} sending to P0 2 ")
                            comm.Send(data, dest=0, tag=2*ipair+1)
                            printMPI(f"Rank {rank} done sending to P0 2")
                            next_pair += skip_pairs

                    if rank == 0:
                        if nprocs > 1:
                                skip_pairs_remotes = nprocs-1
                                remote = ipair % skip_pairs_remotes + 1
                                
                                ant = np.empty(1, dtype=np.int32)
                                printMPI(f"P0 getting from remote {remote} 1")
                                comm.Recv(ant, source=remote, tag=2*ipair)
                                printMPI(f"P0 done getting from remote {remote} 1")
                                nt = ant[0]
                                data = np.empty((nt,4), dtype=np.float64)
                                printMPI(f"P0 getting from remote {remote} 2")
                                comm.Recv(data, source=remote, tag=2*ipair+1)
                                printMPI(f"P0 done getting from remote {remote} 2")
                                z_stf = data[:,0]
                                e_stf = data[:,1]
                                n_stf = data[:,2]
                                t = data[:,3]    
                        next_pair += 1
                        station.add_to_response(z_stf, e_stf, n_stf, t)
                else: 
                    pass
                ipair += 1

            if verbose:
                print(f'ShakerMaker.run - finished my station {i_station} -->  (rank={rank} ipair={ipair} next_pair={next_pair})')
            self._logger.debug(f'ShakerMaker.run - finished station {i_station} (rank={rank} ipair={ipair} next_pair={next_pair})')

            if writer and rank == 0:
                printMPI(f"Rank 0 is writing station {i_station}")
                writer.write_station(station, i_station)
                printMPI(f"Rank 0 is done writing station {i_station}")

        if writer and rank == 0:
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

    def _call_core(self, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        mb = crust.nlayers

        if verbose:
            print("_call_core")
            # print(f"        psource = {psource}")
            print(f"        psource.x = {psource.x}")
            # print(f"        station = {station}")
            print(f"        station.x = {station.x}")

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
        if verbose:
            print('ShakerMaker._call_core - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
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
