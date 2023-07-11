import copy
import numpy as np
import logging
from shakermaker.crustmodel import CrustModel
from shakermaker.faultsource import FaultSource
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker import core 
import imp
import traceback
from time import perf_counter


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
        debugMPI=False,
        tmin=0.,
        tmax=100,
        showProgress=True
        ):
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
        title = f"ShakerMaker Run begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        
        if rank == 0:
            print("\n\n")
            print(title)
            print("-"*len(title))

        #Initialize performance counters
        perf_time_begin = perf_counter()

        perf_time_core = np.zeros(1,dtype=np.double)
        perf_time_send = np.zeros(1,dtype=np.double)
        perf_time_recv = np.zeros(1,dtype=np.double)
        perf_time_conv = np.zeros(1,dtype=np.double)
        perf_time_add = np.zeros(1,dtype=np.double)

        if debugMPI:
            # printMPI = lambda *args : print(*args)
            fid_debug_mpi = open(f"rank_{rank}.debuginfo","w")
            def printMPI(*args):
                fid_debug_mpi.write(*args)
                fid_debug_mpi.write("\n")

        else:
            import os
            fid_debug_mpi = open(os.devnull,"w")
            printMPI = lambda *args : None

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
                        t1 = perf_counter()
                        tdata, z, e, n, t0 = self._call_core(dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                             taper, aux_crust, psource, station, verbose)
                        t2 = perf_counter()
                        perf_time_core += t2 - t1
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        dd = psource.x - station.x
                        dh = np.sqrt(dd[0]**2 + dd[1]**2)
                        dz = np.abs(dd[2])
                        print(f" *** {ipair} {psource.tt=} {t0[0]=} {dh=} {dz=}")


                        t1 = perf_counter()
                        t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                        psource.stf.dt = dt


                        z_stf = psource.stf.convolve(z, t)
                        e_stf = psource.stf.convolve(e, t)
                        n_stf = psource.stf.convolve(n, t)
                        t2 = perf_counter()
                        perf_time_conv += t2 - t1


                        if rank > 0:
                            t1 = perf_counter()
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
                            t2 = perf_counter()
                            perf_time_send += t2 - t1

                    if rank == 0:
                        if nprocs > 1:
                                skip_pairs_remotes = nprocs-1
                                remote = ipair % skip_pairs_remotes + 1

                                t1 = perf_counter()

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

                                t2 = perf_counter()
                                perf_time_recv += t2 - t1
                        next_pair += 1
                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf, t, tmin, tmax)
                            t2 = perf_counter()
                            perf_time_add += t2 - t1
                        except:
                            traceback.print_exc()

                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress:
                            print(f"{ipair} of {npairs} done {t[0]=:0.4f} {t[-1]=:0.4f} ({tmin=:0.4f} {tmax=:0.4f})")

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

        fid_debug_mpi.close()

        perf_time_end = perf_counter()

        if rank == 0 and use_mpi:
            perf_time_total = perf_time_end - perf_time_begin

            print("\n\n")
            print(f"ShakerMaker Run done. Total time: {perf_time_total} s")
            print("------------------------------------------------")

        if use_mpi and nprocs > 1:
            all_max_perf_time_core = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_send = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_recv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_conv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_add = np.array([-np.infty],dtype=np.double)

            all_min_perf_time_core = np.array([np.infty],dtype=np.double)
            all_min_perf_time_send = np.array([np.infty],dtype=np.double)
            all_min_perf_time_recv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_conv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_add = np.array([np.infty],dtype=np.double)

            # Gather statistics from all processes

            comm.Reduce(perf_time_core,
                all_max_perf_time_core, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_send,
                all_max_perf_time_send, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_recv,
                all_max_perf_time_recv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_conv,
                all_max_perf_time_conv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_add,
                all_max_perf_time_add, op = MPI.MAX, root = 0)

            comm.Reduce(perf_time_core,
                all_min_perf_time_core, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_send,
                all_min_perf_time_send, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_recv,
                all_min_perf_time_recv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_conv,
                all_min_perf_time_conv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_add,
                all_min_perf_time_add, op = MPI.MIN, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_max_perf_time_core, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_max_perf_time_send, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_max_perf_time_recv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_max_perf_time_conv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_max_perf_time_add, MPI.DOUBLE], op = MPI.MAX, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_min_perf_time_core, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_min_perf_time_send, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_min_perf_time_recv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_min_perf_time_conv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_min_perf_time_add, MPI.DOUBLE], op = MPI.MIN, root = 0)

            if rank == 0:
                print("\n")
                print("Performance statistics for all processes")
                print(f"time_core     :  max: {all_max_perf_time_core[0]} ({all_max_perf_time_core[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_core[0]} ({all_min_perf_time_core[0]/perf_time_total*100:0.3f}%)")
                print(f"time_send     :  max: {all_max_perf_time_send[0]} ({all_max_perf_time_send[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_send[0]} ({all_min_perf_time_send[0]/perf_time_total*100:0.3f}%)")
                print(f"time_recv     :  max: {all_max_perf_time_recv[0]} ({all_max_perf_time_recv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_recv[0]} ({all_min_perf_time_recv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_conv :  max: {all_max_perf_time_conv[0]} ({all_max_perf_time_conv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_conv[0]} ({all_min_perf_time_conv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_add      :  max: {all_max_perf_time_add[0]} ({all_max_perf_time_add[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_add[0]} ({all_min_perf_time_add[0]/perf_time_total*100:0.3f}%)")



    def run_fast(self, 
        tdata_database_name,
        delta_h=0.02,
        delta_v=0.001,
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
        debugMPI=False,
        tmin=0.,
        tmax=100,
        showProgress=True
        ):
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
        title = f"ShakerMaker Run Fase begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        
        if rank==0:
            print(f"Will write Green's functions database to : {tdata_database_name}")
        data_pairs = np.load(tdata_database_name, allow_pickle=True)

        dists= data_pairs["dists"]
        pairs_to_compute = data_pairs["pairs_to_compute"]
        dh_of_pairs = data_pairs["dh_of_pairs"]
        dv_of_pairs = data_pairs["dv_of_pairs"]
        zrec_of_pairs= data_pairs["zrec_of_pairs"]

        npairs = len(dh_of_pairs)

        # tdata_dict = np.load(tdata_database_name)[0][()]
        tdata_dict = data_pairs["tdata_dict"][()]  #Database with all green's functions


        if rank == 0:
            print("\n\n")
            print(title)
            print("-"*len(title))

        #Initialize performance counters
        perf_time_begin = perf_counter()

        perf_time_core = np.zeros(1,dtype=np.double)
        perf_time_send = np.zeros(1,dtype=np.double)
        perf_time_recv = np.zeros(1,dtype=np.double)
        perf_time_conv = np.zeros(1,dtype=np.double)
        perf_time_add = np.zeros(1,dtype=np.double)

        if debugMPI:
            # printMPI = lambda *args : print(*args)
            fid_debug_mpi = open(f"rank_{rank}.debuginfo","w")
            def printMPI(*args):
                fid_debug_mpi.write(*args)
                fid_debug_mpi.write("\n")

        else:
            import os
            fid_debug_mpi = open(os.devnull,"w")
            printMPI = lambda *args : None

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


                        x_src = psource.x
                        x_rec = station.x
                    
                        z_rec = station.x[2]

                        d = x_rec - x_src
                        dh = np.sqrt(np.dot(d[0:2],d[0:2]))
                        dv = np.abs(d[2])

                        dists[ipair,0] = dh
                        dists[ipair,1] = dv

                        # Get the target Green's Functions
                        ipair_target = 0
                        
                        # print(f"{dh_of_pairs=}")
                        # print(f"{dh_of_pairs.shape=}")
                        # print(f"{dv_of_pairs=}")
                        # print(f"{dv_of_pairs.shape=}")
                        # print(f"{zrec_of_pairs=}")
                        # print(f"{zrec_of_pairs.shape=}")
                        for i in range(len(dh_of_pairs)):
                            dh_p, dv_p, zrec_p = dh_of_pairs[i], dv_of_pairs[i], zrec_of_pairs[i]
                            if abs(dh - dh_p) < delta_h and \
                                abs(dv - dv_p) < delta_v and \
                                abs(z_rec - zrec_p) < delta_v:
                                break
                            else:
                                ipair_target += 1

                        if ipair_target == len(dh_of_pairs):
                            print("Target not found in database")

                        tdata = tdata_dict[ipair_target]

                        if verbose:
                            print("calling core START")
                        t1 = perf_counter()
                        z, e, n, t0 = self._call_core_fast(tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                             taper, aux_crust, psource, station, verbose)
                        t2 = perf_counter()
                        perf_time_core += t2 - t1
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        dd = psource.x - station.x
                        dh = np.sqrt(dd[0]**2 + dd[1]**2)
                        dz = np.abs(dd[2])
                        print(f" *** {ipair} {psource.tt=} {t0[0]=} {dh=} {dz=}")


                        t1 = perf_counter()
                        t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                        psource.stf.dt = dt


                        z_stf = psource.stf.convolve(z, t)
                        e_stf = psource.stf.convolve(e, t)
                        n_stf = psource.stf.convolve(n, t)
                        t2 = perf_counter()
                        perf_time_conv += t2 - t1


                        if rank > 0:
                            t1 = perf_counter()
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
                            t2 = perf_counter()
                            perf_time_send += t2 - t1

                    if rank == 0:
                        if nprocs > 1:
                                skip_pairs_remotes = nprocs-1
                                remote = ipair % skip_pairs_remotes + 1

                                t1 = perf_counter()

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

                                t2 = perf_counter()
                                perf_time_recv += t2 - t1
                        next_pair += 1
                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf, t, tmin, tmax)
                            t2 = perf_counter()
                            perf_time_add += t2 - t1
                        except:
                            traceback.print_exc()

                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress:
                            print(f"{ipair} of {npairs} done {t[0]=:0.4f} {t[-1]=:0.4f} ({tmin=:0.4f} {tmax=:0.4f})")

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

        fid_debug_mpi.close()

        perf_time_end = perf_counter()

        if rank == 0 and use_mpi:
            perf_time_total = perf_time_end - perf_time_begin

            print("\n\n")
            print(f"ShakerMaker Run done. Total time: {perf_time_total} s")
            print("------------------------------------------------")

        if use_mpi and nprocs > 1:
            all_max_perf_time_core = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_send = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_recv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_conv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_add = np.array([-np.infty],dtype=np.double)

            all_min_perf_time_core = np.array([np.infty],dtype=np.double)
            all_min_perf_time_send = np.array([np.infty],dtype=np.double)
            all_min_perf_time_recv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_conv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_add = np.array([np.infty],dtype=np.double)

            # Gather statistics from all processes

            comm.Reduce(perf_time_core,
                all_max_perf_time_core, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_send,
                all_max_perf_time_send, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_recv,
                all_max_perf_time_recv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_conv,
                all_max_perf_time_conv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_add,
                all_max_perf_time_add, op = MPI.MAX, root = 0)

            comm.Reduce(perf_time_core,
                all_min_perf_time_core, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_send,
                all_min_perf_time_send, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_recv,
                all_min_perf_time_recv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_conv,
                all_min_perf_time_conv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_add,
                all_min_perf_time_add, op = MPI.MIN, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_max_perf_time_core, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_max_perf_time_send, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_max_perf_time_recv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_max_perf_time_conv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_max_perf_time_add, MPI.DOUBLE], op = MPI.MAX, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_min_perf_time_core, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_min_perf_time_send, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_min_perf_time_recv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_min_perf_time_conv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_min_perf_time_add, MPI.DOUBLE], op = MPI.MIN, root = 0)

            if rank == 0:
                print("\n")
                print("Performance statistics for all processes")
                print(f"time_core     :  max: {all_max_perf_time_core[0]} ({all_max_perf_time_core[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_core[0]} ({all_min_perf_time_core[0]/perf_time_total*100:0.3f}%)")
                print(f"time_send     :  max: {all_max_perf_time_send[0]} ({all_max_perf_time_send[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_send[0]} ({all_min_perf_time_send[0]/perf_time_total*100:0.3f}%)")
                print(f"time_recv     :  max: {all_max_perf_time_recv[0]} ({all_max_perf_time_recv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_recv[0]} ({all_min_perf_time_recv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_conv :  max: {all_max_perf_time_conv[0]} ({all_max_perf_time_conv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_conv[0]} ({all_min_perf_time_conv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_add      :  max: {all_max_perf_time_add[0]} ({all_max_perf_time_add[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_add[0]} ({all_min_perf_time_add[0]/perf_time_total*100:0.3f}%)")







    def run_create_greens_function_database(self, 
        tdata_database_name,
        pairs_database_name,
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
        debugMPI=False,
        tmin=0.,
        tmax=100,
        showProgress=True
        ):
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
        title = f"ShakerMaker Gen Green's functions database begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        

        if rank==0:
            print(f"Loading pairs-to-compute info from file: {pairs_database_name}")
            print(f"Will write Green's functions database to : {tdata_database_name}")
        data_pairs = np.load(pairs_database_name)

        dists= data_pairs["dists"]
        pairs_to_compute = data_pairs["pairs_to_compute"]
        dh_of_pairs = data_pairs["dh_of_pairs"]
        dv_of_pairs = data_pairs["dv_of_pairs"]
        zrec_of_pairs= data_pairs["zrec_of_pairs"]

        npairs = len(dh_of_pairs)

        if rank == 0:
            print("\n\n")
            print(title)
            print("-"*len(title))

        #Initialize performance counters
        perf_time_begin = perf_counter()

        perf_time_core = np.zeros(1,dtype=np.double)
        perf_time_send = np.zeros(1,dtype=np.double)
        perf_time_recv = np.zeros(1,dtype=np.double)
        perf_time_conv = np.zeros(1,dtype=np.double)
        perf_time_add = np.zeros(1,dtype=np.double)

        if debugMPI:
            # printMPI = lambda *args : print(*args)
            fid_debug_mpi = open(f"rank_{rank}.debuginfo","w")
            def printMPI(*args):
                fid_debug_mpi.write(*args)
                fid_debug_mpi.write("\n")

        else:
            import os
            fid_debug_mpi = open(os.devnull,"w")
            printMPI = lambda *args : None

        self._logger.info('ShakerMaker.run_create_greens_function_database - starting\n\tNumber of sources: {}\n\tNumber of receivers: {}\n'
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

        if rank == 0:
            tdata_dict = {}


        # npairs = self._receivers.nstations*len(self._source._pslist)

        # for i_station, station in enumerate(self._receivers):
        #     for i_psource, psource in enumerate(self._source):
        if True:
            for i_station, i_psource in pairs_to_compute:
                aux_crust = copy.deepcopy(self._crust)

                station = self._receivers.get_station_by_id(i_station)
                psource = self._source.get_source_by_id(i_psource)

                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])


                if ipair == next_pair:
                    if verbose:
                        print(f"rank={rank} nprocs={nprocs} ipair={ipair} skip_pairs={skip_pairs} npairs={npairs} !!")
                    if nprocs == 1 or (rank > 0 and nprocs > 1):

                        if verbose:
                            print("calling core START")
                        t1 = perf_counter()
                        tdata, z, e, n, t0 = self._call_core(dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                             taper, aux_crust, psource, station, verbose)
                        t2 = perf_counter()
                        perf_time_core += t2 - t1
                        if verbose:
                            print("calling core END")

                        nt = len(z)
                        dd = psource.x - station.x
                        dh = np.sqrt(dd[0]**2 + dd[1]**2)
                        dz = np.abs(dd[2])
                        z_rec = station.x[2]
                        print(f" *** {ipair} {psource.tt=} {t0[0]=} {dh=} {dz=}")


                        t1 = perf_counter()
                        # t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                        t = np.array([t0])
                        psource.stf.dt = dt


                        # z_stf = psource.stf.convolve(z, t)
                        # e_stf = psource.stf.convolve(e, t)
                        # n_stf = psource.stf.convolve(n, t)
                        t2 = perf_counter()
                        perf_time_conv += t2 - t1


                        if rank > 0:
                            t1 = perf_counter()
                            ant = np.array([nt], dtype=np.int32).copy()
                            printMPI(f"Rank {rank} sending to P0 1")
                            comm.Send(ant, dest=0, tag=2*ipair)
                            comm.Send(t, dest=0, tag=2*ipair+1)
                            # data = np.empty((nt,4), dtype=np.float64)
                            printMPI(f"Rank {rank} done sending to P0 1")
                            # data[:,0] = z_stf
                            # data[:,1] = e_stf
                            # data[:,2] = n_stf
                            # data[:,3] = t
                            printMPI(f"Rank {rank} sending to P0 2 ")
                            comm.Send(tdata, dest=0, tag=2*ipair+2)
                            printMPI(f"Rank {rank} done sending to P0 2")
                            next_pair += skip_pairs
                            t2 = perf_counter()
                            perf_time_send += t2 - t1

                    if rank == 0:
                        if nprocs > 1:
                                skip_pairs_remotes = nprocs-1
                                remote = ipair % skip_pairs_remotes + 1

                                t1 = perf_counter()

                                ant = np.empty(1, dtype=np.int32)
                                t = np.empty(1, dtype=np.double)
                                printMPI(f"P0 getting from remote {remote} 1")
                                comm.Recv(ant, source=remote, tag=2*ipair)
                                comm.Recv(t, source=remote, tag=2*ipair+1)
                                printMPI(f"P0 done getting from remote {remote} 1")
                                nt = ant[0]

                                tdata = np.empty((nt,9), dtype=np.float64)
                                printMPI(f"P0 getting from remote {remote} 2")
                                comm.Recv(tdata, source=remote, tag=2*ipair+2)
                                printMPI(f"P0 done getting from remote {remote} 2")
                                # z_stf = data[:,0]
                                # e_stf = data[:,1]
                                # n_stf = data[:,2]
                                # t = data[:,3]    
                                dd = psource.x - station.x
                                dh = np.sqrt(dd[0]**2 + dd[1]**2)
                                dz = np.abs(dd[2])
                                z_rec = station.x[2]
                                tdata_dict[ipair] = (t[0], i_station, i_psource, tdata, dh,dz,z_rec)

                                t2 = perf_counter()
                                perf_time_recv += t2 - t1
                        next_pair += 1

                        if showProgress:
                            print(f"{ipair} of {npairs} done")# t[0]={t[0]:0.4f} t[-1]={t[-1]:0.4f} (tmin={tmin:0.4f}, tmax={tmax:0.4f})")


                else: 
                    pass
                ipair += 1

            if verbose:
                print(f'ShakerMaker.run_create_greens_function_database - finished my station {i_station} -->  (rank={rank} ipair={ipair} next_pair={next_pair})')
            self._logger.debug(f'ShakerMaker.run_create_greens_function_database - finished station {i_station} (rank={rank} ipair={ipair} next_pair={next_pair})')

            if writer and rank == 0:
                printMPI(f"Rank 0 is writing station {i_station}")
                writer.write_station(station, i_station)
                printMPI(f"Rank 0 is done writing station {i_station}")

        if rank == 0:
            np.savez(tdata_database_name, tdata_dict=tdata_dict, 
                dists=dists,
                pairs_to_compute=pairs_to_compute,
                dh_of_pairs=dh_of_pairs,
                dv_of_pairs=dv_of_pairs,
                zrec_of_pairs=zrec_of_pairs,)

        fid_debug_mpi.close()

        perf_time_end = perf_counter()

        if rank == 0 and use_mpi:
            perf_time_total = perf_time_end - perf_time_begin

            print("\n\n")
            print(f"ShakerMaker Generate GF database done. Total time: {perf_time_total} s")
            print("------------------------------------------------")

        if use_mpi and nprocs > 1:
            all_max_perf_time_core = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_send = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_recv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_conv = np.array([-np.infty],dtype=np.double)
            all_max_perf_time_add = np.array([-np.infty],dtype=np.double)

            all_min_perf_time_core = np.array([np.infty],dtype=np.double)
            all_min_perf_time_send = np.array([np.infty],dtype=np.double)
            all_min_perf_time_recv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_conv = np.array([np.infty],dtype=np.double)
            all_min_perf_time_add = np.array([np.infty],dtype=np.double)

            # Gather statistics from all processes

            comm.Reduce(perf_time_core,
                all_max_perf_time_core, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_send,
                all_max_perf_time_send, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_recv,
                all_max_perf_time_recv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_conv,
                all_max_perf_time_conv, op = MPI.MAX, root = 0)
            comm.Reduce(perf_time_add,
                all_max_perf_time_add, op = MPI.MAX, root = 0)

            comm.Reduce(perf_time_core,
                all_min_perf_time_core, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_send,
                all_min_perf_time_send, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_recv,
                all_min_perf_time_recv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_conv,
                all_min_perf_time_conv, op = MPI.MIN, root = 0)
            comm.Reduce(perf_time_add,
                all_min_perf_time_add, op = MPI.MIN, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_max_perf_time_core, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_max_perf_time_send, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_max_perf_time_recv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_max_perf_time_conv, MPI.DOUBLE], op = MPI.MAX, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_max_perf_time_add, MPI.DOUBLE], op = MPI.MAX, root = 0)

            # comm.Reduce([np.array([perf_time_core]), MPI.DOUBLE],
            #     [all_min_perf_time_core, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_send]), MPI.DOUBLE],
            #     [all_min_perf_time_send, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_recv]), MPI.DOUBLE],
            #     [all_min_perf_time_recv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_conv]), MPI.DOUBLE],
            #     [all_min_perf_time_conv, MPI.DOUBLE], op = MPI.MIN, root = 0)
            # comm.Reduce([np.array([perf_time_add]), MPI.DOUBLE],
            #     [all_min_perf_time_add, MPI.DOUBLE], op = MPI.MIN, root = 0)

            if rank == 0:
                print("\n")
                print("Performance statistics for all processes")
                print(f"time_core     :  max: {all_max_perf_time_core[0]} ({all_max_perf_time_core[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_core[0]} ({all_min_perf_time_core[0]/perf_time_total*100:0.3f}%)")
                print(f"time_send     :  max: {all_max_perf_time_send[0]} ({all_max_perf_time_send[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_send[0]} ({all_min_perf_time_send[0]/perf_time_total*100:0.3f}%)")
                print(f"time_recv     :  max: {all_max_perf_time_recv[0]} ({all_max_perf_time_recv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_recv[0]} ({all_min_perf_time_recv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_conv :  max: {all_max_perf_time_conv[0]} ({all_max_perf_time_conv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_conv[0]} ({all_min_perf_time_conv[0]/perf_time_total*100:0.3f}%)")
        print(f"time_add      :  max: {all_max_perf_time_add[0]} ({all_max_perf_time_add[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_add[0]} ({all_min_perf_time_add[0]/perf_time_total*100:0.3f}%)")



    def gen_greens_function_database_pairs(self,
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
        debugMPI=False,
        tmin=0.,
        tmax=100,
        delta_h=0.04,
        delta_v=0.002,
        showProgress=True,
        store_here=None,
        ):
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
        title = f"ShakerMaker Gen GF database pairs begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        
        if rank == 0:
            print("\n\n")
            print(title)
            print("-"*len(title))

    
        self._logger.info('ShakerMaker.run - starting\n\tNumber of sources: {}\n\tNumber of receivers: {}\n'
                          '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
                          .format(self._source.nsources, self._receivers.nstations,
                                  self._source.nsources*self._receivers.nstations, dt, nfft))

        ipair = 0
        if nprocs == 1 or rank == 0:
            next_pair = rank
            skip_pairs = 1
        else :
            next_pair = rank-1
            skip_pairs = nprocs-1

        npairs = self._receivers.nstations*len(self._source._pslist)

        print(f"{npairs=}")

        dists = np.zeros((npairs, 2))

        # pairs_to_compute = np.array([] ,dtype=np.int32)
        # pairs_to_compute = []
        # dh_of_pairs = np.array([] ,dtype=np.double)
        # dh_of_pairs = []
        # dv_of_pairs = np.array([] ,dtype=np.double)
        # dv_of_pairs = []
        # zrec_of_pairs = np.array([] ,dtype=np.double)
        # zrec_of_pairs = []
        pairs_to_compute = np.empty((npairs, 2), dtype=np.int32)
        dh_of_pairs = np.empty(npairs, dtype=np.double)
        dv_of_pairs = np.empty(npairs, dtype=np.double)
        zrec_of_pairs = np.empty(npairs, dtype=np.double)

        # Initialize the counter for the number of computed pairs.
        n_computed_pairs = 0

        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):

                x_src = psource.x
                x_rec = station.x
                
                z_rec = station.x[2]

                d = x_rec - x_src

                dh = np.sqrt(np.dot(d[0:2],d[0:2]))
                dv = np.abs(d[2])

                dists[ipair,0] = dh
                dists[ipair,1] = dv
                # if len(dh_of_pairs) == 0:
                #     # pairs_to_compute = np.append(pairs_to_compute,(i_station, i_psource))
                #     pairs_to_compute.append((i_station, i_psource))
                #     # dh_of_pairs = np.append(dh_of_pairs,dh)
                #     dh_of_pairs.append(dh)
                #     # dv_of_pairs = np.append(dv_of_pairs,dv)
                #     dv_of_pairs.append(dv)
                #     # zrec_of_pairs = np.append(zrec_of_pairs,z_rec)
                #     zrec_of_pairs.append(z_rec)


                # for i in range(len(dh_of_pairs)):
                #     dh_p, dv_p, zrec_p = dh_of_pairs[i], dv_of_pairs[i], zrec_of_pairs[i]
                #     if abs(dh - dh_p) < delta_h and \
                #         abs(dv - dv_p) < delta_v and \
                #         abs(z_rec - zrec_p) < delta_v:
                #         pass
                #     else:
                #         # pairs_to_compute = np.append(pairs_to_compute,(i_station, i_psource))
                #         pairs_to_compute.append((i_station, i_psource))
                #         # dh_of_pairs = np.append(dh_of_pairs,dh)
                #         dh_of_pairs.append(dh)
                #         # dv_of_pairs = np.append(dv_of_pairs,dv)
                #         dv_of_pairs.append(dv)
                #         # zrec_of_pairs = np.append(zrec_of_pairs,z_rec)
                #         zrec_of_pairs.append(z_rec)
                condition = (np.abs(dh - dh_of_pairs[:n_computed_pairs]) < delta_h) & \
                            (np.abs(dv - dv_of_pairs[:n_computed_pairs]) < delta_v) & \
                            (np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) < delta_v)

                if n_computed_pairs == 0 or not np.any(condition):
                    pairs_to_compute[n_computed_pairs,:] = [i_station, i_psource]
                    dh_of_pairs[n_computed_pairs] = dh
                    dv_of_pairs[n_computed_pairs] = dv
                    zrec_of_pairs[n_computed_pairs] = z_rec

                    # print(f"Added GF # {n_computed_pairs} for {dh=} {dv=} {z_rec=} ")

                    n_computed_pairs += 1

                if ipair % 10000 == 0:
                    print(f"On {ipair=} of {npairs} {n_computed_pairs=}")

                ipair += 1

        # pairs_to_compute = np.array(pairs_to_compute)
        # dh_of_pairs = np.array(dh_of_pairs)
        # dv_of_pairs = np.array(dv_of_pairs)
        # zrec_of_pairs = np.array(zrec_of_pairs)
        # Slice the arrays to remove unused parts.
        pairs_to_compute = pairs_to_compute[:n_computed_pairs,:]
        dh_of_pairs = dh_of_pairs[:n_computed_pairs]
        dv_of_pairs = dv_of_pairs[:n_computed_pairs]
        zrec_of_pairs = zrec_of_pairs[:n_computed_pairs]

        # pairs_to_compute = pairs_to_compute.reshape((-1,2))

        print(f"Need only {n_computed_pairs} pairs of {npairs} ({n_computed_pairs/npairs*100}% reduction)")



        if store_here is not None:
            np.savez(store_here,
                dists=dists,
                pairs_to_compute=pairs_to_compute,
                dh_of_pairs=dh_of_pairs,
                dv_of_pairs=dv_of_pairs,
                zrec_of_pairs=zrec_of_pairs,
            )

            return dists, pairs_to_compute, dh_of_pairs, dv_of_pairs, zrec_of_pairs

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


    def _call_core_fast(self, tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc, taper, crust, psource, station, verbose=False):
        mb = crust.nlayers

        if verbose:
            print("_call_core_fast")
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

        self._logger.debug('ShakerMaker._call_core_fast - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
                           '\tstyoe: {}\n\tupdn: {}\n\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                           '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n\tsmth: {}\n\twc1: {}\n\twc2: {}\n'
                           '\tpmin: {}\n\tpmax: {}\n\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n\tpf: {}\n\tdf: {}\n'
                           '\tlf: {}\n\tsx: {}\n\tsy: {}\n\trx: {}\n\try: {}\n\t'
                           .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1,
                                   wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))
        if verbose:
            print('ShakerMaker._call_core_fast - calling core.subgreen\n\tmb: {}\n\tsrc: {}\n\trcv: {}\n'
                   '\tstyoe: {}\n\tupdn: {}\n\td: {}\n\ta: {}\n\tb: {}\n\trho: {}\n\tqa: {}\n\tqb: {}\n'
                   '\tdt: {}\n\tnfft: {}\n\ttb: {}\n\tnx: {}\n\tsigma: {}\n\tsmth: {}\n\twc1: {}\n\twc2: {}\n'
                   '\tpmin: {}\n\tpmax: {}\n\tdk: {}\n\tkc: {}\n\ttaper: {}\n\tx: {}\n\tpf: {}\n\tdf: {}\n'
                   '\tlf: {}\n\tsx: {}\n\tsy: {}\n\trx: {}\n\try: {}\n\t'
                   .format(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma, smth, wc1,
                           wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, sx, sy, rx, ry))

        # Execute the core subgreen fortran routing
        tdata_ = tdata[3].T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))
        print(f"{tdata_=}")
        print(f"{tdata_.shape=}")
        z, e, n, t0 = core.subgreen2(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma,
                                           smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        self._logger.debug('ShakerMaker._call_core_fast - core.subgreen returned: z_size'.format(len(z)))

        return z, e, n, t0
