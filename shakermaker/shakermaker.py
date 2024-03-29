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
                        # print(f" *** {ipair} {psource.tt=} {t0[0]=} {dh=} {dz=}")


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
        h5_database_name,
        delta_h=0.04,
        delta_v_rec=0.002,
        delta_v_src=0.2,
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
            print(f"Loading pairs-to-compute info from HDF5 database: {h5_database_name}")

        import h5py

        if rank > 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r')
        elif rank == 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r+')



        # dists= hfile["/dists"][:]
        pairs_to_compute = hfile["/pairs_to_compute"][:]
        dh_of_pairs = hfile["/dh_of_pairs"][:]
        dv_of_pairs = hfile["/dv_of_pairs"][:]
        zrec_of_pairs= hfile["/zrec_of_pairs"][:]
        zsrc_of_pairs= hfile["/zsrc_of_pairs"][:]


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

        tstart = perf_counter()

        npairs = self._receivers.nstations*len(self._source._pslist)
        npairs_skip  = 0
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
                    
                        z_src = psource.x[2]
                        z_rec = station.x[2]

                        d = x_rec - x_src
                        dh = np.sqrt(np.dot(d[0:2],d[0:2]))
                        dv = np.abs(d[2])

                        # dists[ipair,0] = dh
                        # dists[ipair,1] = dv

                        # Get the target Green's Functions
                        ipair_target = 0
                        # condition = lor(np.abs(dh - dh_of_pairs[:n_computed_pairs])      > delta_h,     \
                                        # np.abs(z_src - zsrc_of_pairs[:n_computed_pairs]) > delta_v_src, \
                                        # np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) > delta_v_rec)
                        for i in range(len(dh_of_pairs)):
                            dh_p, dv_p, zrec_p, zsrc_p = dh_of_pairs[i], dv_of_pairs[i], zrec_of_pairs[i], zsrc_of_pairs[i]
                            if abs(dh - dh_p) < delta_h and \
                                abs(z_src - zsrc_p) < delta_v_src and \
                                abs(z_rec - zrec_p) < delta_v_rec:
                                break
                            else:
                                ipair_target += 1

                        if ipair_target == len(dh_of_pairs):
                            print("Target not found in database -- SKIPPING")
                            npairs_skip += 1
                            if npairs_skip > 500:
                                print(f"Rank {rank} skipped too many pairs, giving up!")
                                exit(-1)
                                break
                            else:
                                continue

                        # tdata = tdata_dict[ipair_target]
                        ipair_string = "/tdata_dict/"+str(ipair_target)+"_tdata"
                        # print(f"Looking in database for {ipair_string}")
                        tdata = hfile[ipair_string][:]

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
                        # print(f" *** {ipair} {psource.tt=} {t0[0]=} {dh=} {dz=}")


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
                            progress_percent = ipair/npairs*100
                            tnow = perf_counter()

                            time_per_pair = (tnow - tstart)/(ipair+1)

                            time_left = (npairs - ipair - 1)*time_per_pair

                            hh = np.floor(time_left / 3600)
                            mm = np.floor((time_left - hh*3600)/60)
                            ss = time_left - mm*60 - hh*3600

                            print(f"{ipair} of {npairs} ({progress_percent:.4f}%) ETA = {hh:.0f}:{mm:.0f}:{ss:.1f} {t[0]=:0.4f} {t[-1]=:0.4f} ({tmin=:0.4f} {tmax=:0.4f})")

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

            if rank == 0:
                print("\n")
                print("Performance statistics for all processes")
                print(f"time_core     :  max: {all_max_perf_time_core[0]} ({all_max_perf_time_core[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_core[0]} ({all_min_perf_time_core[0]/perf_time_total*100:0.3f}%)")
                print(f"time_send     :  max: {all_max_perf_time_send[0]} ({all_max_perf_time_send[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_send[0]} ({all_min_perf_time_send[0]/perf_time_total*100:0.3f}%)")
                print(f"time_recv     :  max: {all_max_perf_time_recv[0]} ({all_max_perf_time_recv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_recv[0]} ({all_min_perf_time_recv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_conv :  max: {all_max_perf_time_conv[0]} ({all_max_perf_time_conv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_conv[0]} ({all_min_perf_time_conv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_add      :  max: {all_max_perf_time_add[0]} ({all_max_perf_time_add[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_add[0]} ({all_min_perf_time_add[0]/perf_time_total*100:0.3f}%)")



    def run_faster(self, 
        h5_database_name,
        delta_h=0.04,
        delta_v_rec=0.002,
        delta_v_src=0.2,
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
        showProgress=True,
        allow_out_of_bounds=False,
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
        import h5py
        # from scipy.spatial import KDTree
        title = f"ShakerMaker Run Fase begin. {dt=} {nfft=} {dk=} {tb=} {tmin=} {tmax=}"
        

        if rank==0:
            print(f"Loading pairs-to-compute info from HDF5 database: {h5_database_name}")


        if rank > 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r')
        elif rank == 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r+')



        # dists= hfile["/dists"][:]
        pairs_to_compute = hfile["/pairs_to_compute"][:]
        dh_of_pairs = hfile["/dh_of_pairs"][:]
        # dv_of_pairs = hfile["/dv_of_pairs"][:]
        zrec_of_pairs= hfile["/zrec_of_pairs"][:]
        zsrc_of_pairs= hfile["/zsrc_of_pairs"][:]

        #Initialize seach KD-tree:
        # Construct the data for KD-tree
        # data_for_tree = list(zip(dh_of_pairs, zrec_of_pairs, zsrc_of_pairs))
        # tree = KDTree(data_for_tree)

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


        next_station = rank
        skip_stations = nprocs

        tstart = perf_counter()

        #Stage one! Compute each station at each processor, no comm.
        nsources = self._source.nsources
        nstations = self._receivers.nstations
        npairs = nsources*nstations

        npairs_skip  = 0
        ipair = 0

        n_my_stations = 0

        for i_station, station in enumerate(self._receivers):

            tstart_source = perf_counter()
            for i_psource, psource in enumerate(self._source):
                aux_crust = copy.deepcopy(self._crust)

                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])

                if i_station == next_station:

                    if verbose:
                        print(f"{rank=} {nprocs=} {i_station=} {skip_stations=} {npairs=} !!")
                    if True:  #All processors do this always all the time... 
                        x_src = psource.x
                        x_rec = station.x
                    
                        z_src = psource.x[2]
                        z_rec = station.x[2]

                        d = x_rec - x_src
                        dh = np.sqrt(np.dot(d[0:2],d[0:2]))
                        # dv = np.abs(d[2])

                        # Get the target Green's Functions
                        # OLD APPROACH ======================================================================================
                        min_distance = float('inf')
                        best_match_index = -1

                        for i in range(len(dh_of_pairs)):
                            dh_p, zrec_p, zsrc_p = dh_of_pairs[i], zrec_of_pairs[i], zsrc_of_pairs[i]
                            
                            # Check if the current pair is within the tolerances
                            if (abs(dh - dh_p) < delta_h and \
                               abs(z_src - zsrc_p) < delta_v_src and \
                               abs(z_rec - zrec_p) < delta_v_rec) or \
                               allow_out_of_bounds:

                                distance = (abs(dh - dh_p) + 
                                            abs(z_src - zsrc_p) + 
                                            abs(z_rec - zrec_p))
                            
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match_index = i

                        if best_match_index != -1:
                            # Use best_match_index as your best match within the tolerances
                            ipair_target = best_match_index
                        else:
                            print(f"No suitable match found! {allow_out_of_bounds=} {min_distance=}")

                        if ipair_target == len(dh_of_pairs):
                            print("Target not found in database -- SKIPPING")
                            npairs_skip += 1
                            if npairs_skip > 500:
                                print(f"Rank {rank} skipped too many pairs, giving up!")
                                exit(-1)
                                break
                            else:
                                continue
                        # END OLD APPROACH ==================================================================================
                        
                        # NEW APPROACH ======================================================================================
                        #New approach using KD-tree
                        # Query for the current point
                        # point = [dh, z_rec, z_src]
                        # distance, best_match_index = tree.query(point)
                        # # condition = lor(np.abs(dh - dh_of_pairs[:n_computed_pairs])      > delta_h,     \
                        # #                 np.abs(z_src - zsrc_of_pairs[:n_computed_pairs]) > delta_v_src, \
                        # #                 np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) > delta_v_rec)
                        # # Check if the closest match is within the tolerances
                        # if abs(dh - dh_of_pairs[best_match_index]) < delta_h and \
                        #    abs(z_src - zsrc_of_pairs[best_match_index]) < delta_v_src and \
                        #    abs(z_rec - zrec_of_pairs[best_match_index]) < delta_v_rec:
                        #     ipair_target = best_match_index
                        # else:
                        #     print(f"No suitable match found! {best_match_index=}")
                        #     print(f"{abs(dh - dh_of_pairs[best_match_index])=} < {delta_h=} -> {abs(dh - dh_of_pairs[best_match_index]) < delta_h}" )
                        #     print(f"{abs(z_src - zsrc_of_pairs[best_match_index])=} < {delta_v_src=} -> {abs(z_src - zsrc_of_pairs[best_match_index]) < delta_v_src}" )
                        #     print(f"{abs(z_rec - zrec_of_pairs[best_match_index])=} < {delta_v_rec=} -> {abs(z_rec - zrec_of_pairs[best_match_index]) < delta_v_rec}" )
                        # END NEW APPROACH ==================================================================================



                        ipair_string = "/tdata_dict/"+str(ipair_target)+"_tdata"
                        tdata = hfile[ipair_string][:]

                        if verbose:
                            print("calling core FASTER START")
                        t1 = perf_counter()
                        z, e, n, t0 = self._call_core_fast(tdata, dt, nfft, tb, nx, sigma, smth, wc1, wc2, pmin, pmax, dk, kc,
                                                             taper, aux_crust, psource, station, verbose)
                        t2 = perf_counter()
                        perf_time_core += t2 - t1
                        if verbose:
                            print("calling core FASTER END")

                        t1 = perf_counter()
                        t = np.arange(0, len(z)*dt, dt) + psource.tt + t0
                        psource.stf.dt = dt

                        z_stf = psource.stf.convolve(z, t)
                        e_stf = psource.stf.convolve(e, t)
                        n_stf = psource.stf.convolve(n, t)
                        t2 = perf_counter()
                        perf_time_conv += t2 - t1

                        try:
                            t1 = perf_counter()
                            station.add_to_response(z_stf, e_stf, n_stf, t, tmin, tmax)
                            n_my_stations += 1
                            t2 = perf_counter()
                            perf_time_add += t2 - t1
                        except:
                            traceback.print_exc()

                            if use_mpi and nprocs > 1:
                                comm.Abort()

                        if showProgress and rank == 0:
                            #report progress to screen
                            progress_percent = i_psource/nsources*100

                            tnow = perf_counter()

                            time_per_source = (tnow - tstart_source)/(i_psource+1) 

                            time_left = (nsources - i_psource - 1)*time_per_source

                            hh = np.floor(time_left / 3600)
                            mm = np.floor((time_left - hh*3600)/60)
                            ss = time_left - mm*60 - hh*3600

                            if i_psource % 1000 == 0:
                                print(f"   ! RANK {rank} Station {i_station} progress: {i_psource} of {nsources} ({progress_percent:.4f}%) ETA = {hh:.0f}:{mm:02.0f}:{ss:02.1f} {t[0]=:0.4f} {t[-1]=:0.4f}")# ({tmin=:0.4f} {tmax=:0.4f})")
                else:  #if i_station == next_station:
                    pass
                ipair += 1
            if verbose:
                print(f'ShakerMaker.run - finished my station {i_station} -->  ({rank=} {ipair=} {next_station=})')
            self._logger.debug(f'ShakerMaker.run - finished station {i_station} ({rank=} {ipair=} {next_station=})')
            
            if i_station == next_station:

                progress_percent = i_station/nstations*100
                tnow = perf_counter()

                time_per_station = (tnow - tstart_source)
                nstations_left_this_rank = (nstations - i_station - 1)//skip_stations
                time_left = nstations_left_this_rank*time_per_station

                hh = np.floor(time_left / 3600)
                mm = np.floor((time_left - hh*3600)/60)
                ss = time_left - mm*60 - hh*3600

                print(f"{rank=} at {i_station=} of {nstations} ({progress_percent:.4f}%) ETA = {hh:.0f}:{mm:02.0f}:{ss:03.1f}")# {t[0]=:0.4f} {t[-1]=:0.4f} ({tmin=:0.4f} {tmax=:0.4f})")

                next_station += skip_stations
        
        #Stage two! Collect all results at P0

        #First all ranks other than 0 send their stuff to P0
        if rank > 0:
            next_station = rank
            skip_stations = nprocs
            print(f"Rank {rank} is sending its data.")
            for i_station, station in enumerate(self._receivers):
                if i_station == next_station:
                    z,e,n,t = station.get_response()

                    #send to P0
                    t1 = perf_counter()
                    ant = np.array([len(z)], dtype=np.int32).copy()
                    printMPI(f"Rank {rank} sending to P0 1")
                    comm.Send(ant, dest=0, tag=2*i_station)
                    data = np.empty((len(z),4), dtype=np.float64)
                    printMPI(f"Rank {rank} done sending to P0 1")
                    data[:,0] = z
                    data[:,1] = e
                    data[:,2] = n
                    data[:,3] = t
                    printMPI(f"Rank {rank} sending to P0 2 ")
                    comm.Send(data, dest=0, tag=2*i_station+1)
                    printMPI(f"Rank {rank} done sending to P0 2")
                    t2 = perf_counter()
                    perf_time_send += t2 - t1

                    next_station += skip_stations

            print(f"Rank {rank} is DONE sending its data.")


        #Rank 0 recieves all the stuff
        if rank == 0:
            print("Rank 0 is gathering all the results and writing them to disk")
            count_stations = 0
            for remote_rank in range(1,nprocs):
                next_station = remote_rank
                skip_stations = nprocs
                for i_station, station in enumerate(self._receivers):
                    if i_station == next_station:
                        #get from remote
                        t1 = perf_counter()
                        ant = np.empty(1, dtype=np.int32)
                        printMPI(f"P0 getting from remote {i_station} 1")
                        comm.Recv(ant, source=remote_rank, tag=2*i_station)
                        printMPI(f"P0 done getting from remote {i_station} 1")
                        nt = ant[0]
                        data = np.empty((nt,4), dtype=np.float64)
                        printMPI(f"P0 getting from remote {i_station} 2")
                        comm.Recv(data, source=remote_rank, tag=2*i_station+1)
                        printMPI(f"P0 done getting from remote {i_station} 2")
                        z = data[:,0]
                        e = data[:,1]
                        n = data[:,2]
                        t = data[:,3]

                        t2 = perf_counter()
                        perf_time_recv += t2 - t1

                        station.add_to_response(z, e, n, t, tmin, tmax)

                        if writer:
                            printMPI(f"Rank 0 is writing station {i_station}")
                            writer.write_station(station, i_station)
                            printMPI(f"Rank 0 is done writing station {i_station}")

                        next_station += skip_stations
                        count_stations += 1
            print("Rank 0 is DONE gathering ")
            print("Rank 0 writing its own stations ")
            #Now rank 0 writes its own stations
            next_station = 0
            skip_stations = nprocs
            for i_station, station in enumerate(self._receivers):
                if i_station == next_station:
                    if writer:
                        printMPI(f"Rank 0 is writing station {i_station}")
                        writer.write_station(station, i_station)
                        printMPI(f"Rank 0 is done writing station {i_station}")

                    next_station += skip_stations
                    count_stations += 1

            #print accouted for all stations
            count_stations += n_my_stations
            print(f"Rank 0 got {count_stations} of {nstations} stations")
            # assert count_stations == nstations, f"Rank 0 only got {count_stations} of {nstations} stations"

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

            print(f"rank {rank} @ gather all performances stats")

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

            if rank == 0:
                print("\n")
                print("Performance statistics for all processes")
                print(f"time_core     :  max: {all_max_perf_time_core[0]} ({all_max_perf_time_core[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_core[0]} ({all_min_perf_time_core[0]/perf_time_total*100:0.3f}%)")
                print(f"time_send     :  max: {all_max_perf_time_send[0]} ({all_max_perf_time_send[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_send[0]} ({all_min_perf_time_send[0]/perf_time_total*100:0.3f}%)")
                print(f"time_recv     :  max: {all_max_perf_time_recv[0]} ({all_max_perf_time_recv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_recv[0]} ({all_min_perf_time_recv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_conv :  max: {all_max_perf_time_conv[0]} ({all_max_perf_time_conv[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_conv[0]} ({all_min_perf_time_conv[0]/perf_time_total*100:0.3f}%)")
                print(f"time_add      :  max: {all_max_perf_time_add[0]} ({all_max_perf_time_add[0]/perf_time_total*100:0.3f}%) min: {all_min_perf_time_add[0]} ({all_min_perf_time_add[0]/perf_time_total*100:0.3f}%)")






    def run_create_greens_function_database(self, 
        h5_database_name,
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
            print(f"Loading pairs-to-compute info from HDF5 database: {h5_database_name}")

        import h5py

        if rank > 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r')
        elif rank == 0:
            hfile = h5py.File(h5_database_name + '.h5', 'r+')

        pairs_to_compute = hfile["/pairs_to_compute"][:]
        dh_of_pairs = hfile["/dh_of_pairs"][:]
        dv_of_pairs = hfile["/dv_of_pairs"][:]
        zrec_of_pairs= hfile["/zrec_of_pairs"][:]

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

        ipair = 0
        if nprocs == 1 or rank == 0:
            next_pair = rank
            skip_pairs = 1
        else :
            next_pair = rank-1
            skip_pairs = nprocs-1

        if rank == 0:
            tdata_dict = {}

        if rank == 0:
            # Create a group for tdata_dict
            if "tdata_dict" in hfile:
                print("Found TDATA group in the HFILE. Starting anew!")
                del hfile["tdata_dict"]

            tdata_group = hfile.create_group("tdata_dict")


        if rank > 0:
            send_buffers = []
            request_list = []


        if True:
            tstart_pair = perf_counter()
            for i_station, i_psource in pairs_to_compute:
                aux_crust = copy.deepcopy(self._crust)

                station = self._receivers.get_station_by_id(i_station)
                psource = self._source.get_source_by_id(i_psource)

                aux_crust.split_at_depth(psource.x[2])
                aux_crust.split_at_depth(station.x[2])


                if ipair == next_pair:
                    if debugMPI:
                        print(f"{rank=} {nprocs=} {ipair=} {skip_pairs=} {npairs=} !!")
                    if nprocs == 1 or (rank > 0 and nprocs > 1):
                        # print(f"     {rank=} {nprocs=} {ipair=} {skip_pairs=} {npairs=} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")

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

                        t1 = perf_counter()
                        t = np.array([t0])
                        psource.stf.dt = dt

                        t2 = perf_counter()
                        perf_time_conv += t2 - t1


                        if rank > 0:
                            t1 = perf_counter()
                            ant = np.array([nt], dtype=np.int32).copy()
                            printMPI(f"Rank {rank} sending to P0 1")
                            comm.Send(ant, dest=0, tag=3*ipair)
                            comm.Send(t, dest=0, tag=3*ipair+1)
                            printMPI(f"Rank {rank} done sending to P0 1")

                            printMPI(f"Rank {rank} sending to P0 2 ")
                            tdata_c_order = np.empty((nt,9), dtype=np.float64)
                            for comp in range(9):
                                tdata_c_order[:,comp] = tdata[0,comp,:]
                            comm.Send(tdata_c_order, dest=0, tag=3*ipair+2)
                            printMPI(f"Rank {rank} done sending to P0 2")
                            t2 = perf_counter()
                            perf_time_send += t2 - t1
                            next_pair += skip_pairs

                            #Use buffered asynchronous sends
                            # t1 = perf_counter()
                            # # buf = {
                            # #     'ant': np.array([nt], dtype=np.int32).copy(),
                            # #     't': t.copy(),
                            # #     'tdata_c_order': tdata_c_order.copy()
                            # # }
                            # # send_buffers.append(buf)
                            # send_buffers.append(np.array([nt], dtype=np.int32).copy())
                            # request_list.append(comm.Isend(send_buffers[-1], dest=0, tag=3*ipair))
                            # send_buffers.append(t.copy())
                            # request_list.append(comm.Isend(send_buffers[-1], dest=0, tag=3*ipair+1))
                            # send_buffers.append(tdata_c_order.copy())
                            # request_list.append(comm.Isend(send_buffers[-1], dest=0, tag=3*ipair+2))
                            # t2 = perf_counter()
                            # perf_time_send += t2 - t1

                            # print(f"    {rank=} sent {ipair=}")

                            # next_pair += skip_pairs
                            # #Check the completed requests
                            # completed_indices = []
                            # for i_req, request in enumerate(request_list):
                            #     # completed, status = request.Test()
                            #     # item, req = request[0], request[1]
                            #     completed = request.Test()
                            #     if completed:
                            #         completed_indices.append(i_req)

                            # # print(f"{rank=} {completed_indices=}")

                            # try:
                            #     # Remove completed requests and data from buffers
                            #     for i_req in reversed(completed_indices):
                            #         # print(f"{rank=} deleting {i_req=} ")
                            #         del request_list[i_req]
                            #         del send_buffers[i_req]
                            # except:
                            #     print(f"{rank=} failed trying to remove {i_req=} \n{completed_indices=}\n {request_list=}\n {send_buffers=}\n")
                            #     exit(-1)
                            



                    if rank == 0:
                        if nprocs > 1:
                                skip_pairs_remotes = nprocs-1
                                remote = ipair % skip_pairs_remotes + 1

                                t1 = perf_counter()

                                ant = np.empty(1, dtype=np.int32)
                                t = np.empty(1, dtype=np.double)

                                # print(f"{rank=} expecting {ipair=} from {remote=}")

                                printMPI(f"P0 getting from remote {remote} 1")
                                comm.Recv(ant, source=remote, tag=3*ipair)
                                comm.Recv(t, source=remote, tag=3*ipair+1)
                                printMPI(f"P0 done getting from remote {remote} 1")
                                nt = ant[0]

                                tdata = np.empty((nt,9), dtype=np.float64)
                                printMPI(f"P0 getting from remote {remote} 2")
                                comm.Recv(tdata, source=remote, tag=3*ipair+2)
                                printMPI(f"P0 done getting from remote {remote} 2")

                                dd = psource.x - station.x
                                dh = np.sqrt(dd[0]**2 + dd[1]**2)
                                dz = np.abs(dd[2])
                                z_rec = station.x[2]

                                # print(f"{rank=} writing {ipair=} ")
                                tw1 = perf_counter()
                                tdata_group[str(ipair)+"_t0"] = t[0]
                                tdata_group[str(ipair)+"_tdata"] = tdata
                                tw2 = perf_counter()
                                # print(f"{rank=} done writing {ipair=} {tw2-tw1=} ")

                                t2 = perf_counter()
                                perf_time_recv += t2 - t1
                        next_pair += 1

                        if showProgress:
                            tend_pair = perf_counter()

                            time_left = (tend_pair - tstart_pair )*(npairs-ipair)/(ipair+1)

                            hh = np.floor(time_left / 3600)
                            mm = np.floor((time_left - hh*3600)/60)
                            ss = time_left - mm*60 - hh*3600

                            print(f"{ipair} of {npairs} done ETA = {hh:.0f}:{mm:.0f}:{ss:.1f} ")
                #endif

                else: 
                    pass
                ipair += 1

            if verbose:
                print(f'ShakerMaker.run_create_greens_function_database - finished my station {i_station} -->  (rank={rank} ipair={ipair} next_pair={next_pair})')
            self._logger.debug(f'ShakerMaker.run_create_greens_function_database - finished station {i_station} (rank={rank} ipair={ipair} next_pair={next_pair})')

        fid_debug_mpi.close()

        perf_time_end = perf_counter()

        # if rank > 0:
        #     print(f"{rank=} done and waiting for all requests to finish")
        #     for req in request_list:
        #         req.wait()

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
        delta_v_rec=0.002,
        delta_v_src=0.2,
        showProgress=True,
        store_here=None,
        npairs_max=100000,
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
            print(f"Using { delta_h= } { delta_v_rec = } { delta_v_src = }")

    
        self._logger.info('ShakerMaker.run - starting\n\tNumber of sources: {}\n\tNumber of receivers: {}\n'
                          '\tTotal src-rcv pairs: {}\n\tdt: {}\n\tnfft: {}'
                          .format(self._source.nsources, self._receivers.nstations,
                                  self._source.nsources*self._receivers.nstations, dt, nfft))

        ipair = 0

        npairs = self._receivers.nstations*len(self._source._pslist)

        print(f"{npairs=}")

        # dists = np.zeros((npairs, 2))

        pairs_to_compute = np.empty((npairs_max, 2), dtype=np.int32)
        dd_of_pairs = np.empty(npairs_max, dtype=np.double)
        dh_of_pairs = np.empty(npairs_max, dtype=np.double)
        dv_of_pairs = np.empty(npairs_max, dtype=np.double)
        zrec_of_pairs = np.empty(npairs_max, dtype=np.double)
        zsrc_of_pairs = np.empty(npairs_max, dtype=np.double)

        # Initialize the counter for the number of computed pairs.
        n_computed_pairs = 0

        # def lor(a,b,c):
        #     return np.logical_or(a,np.logical_or(b,c))
        def lor(a, b, c):
            return a | b | c


        for i_station, station in enumerate(self._receivers):
            for i_psource, psource in enumerate(self._source):


                if n_computed_pairs >= npairs_max:
                    print("Exceeded number of pairs!!")
                    exit(0)

                t1 = perf_counter()

                x_src = psource.x
                x_rec = station.x
                
                z_rec = station.x[2]
                z_src = psource.x[2]

                d = x_rec - x_src

                dd = np.linalg.norm(d)
                dh = np.linalg.norm(d[0:2])
                dv = np.abs(d[2])

                # dists[ipair,0] = dh
                # dists[ipair,1] = dv
               
                condition = lor(np.abs(dh - dh_of_pairs[:n_computed_pairs])      > delta_h,     \
                                np.abs(z_src - zsrc_of_pairs[:n_computed_pairs]) > delta_v_src, \
                                np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) > delta_v_rec)

                # condition = (np.abs(dd - dd_of_pairs[:n_computed_pairs]) < delta_h) & \
                            # (np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) < delta_v) 
                # condition = (np.abs(dh - dh_of_pairs[:n_computed_pairs]) < delta_h) & \
                #             (np.abs(z_src - zsrc_of_pairs[:n_computed_pairs]) < delta_v_src) &\
                #             (np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) < delta_v_rec) 
                # condition = (np.abs(dh - dh_of_pairs[:n_computed_pairs]) < delta_h) & \
                #             (np.abs(z_rec - zrec_of_pairs[:n_computed_pairs]) < delta_v) & \
                #             (np.abs(z_src - zsrc_of_pairs[:n_computed_pairs]) < delta_v)
                            # (np.abs(dv - dv_of_pairs[:n_computed_pairs]) < delta_v) & \

                if n_computed_pairs == 0 or np.all(condition):
                    pairs_to_compute[n_computed_pairs,:] = [i_station, i_psource]
                    dd_of_pairs[n_computed_pairs] = dd
                    dh_of_pairs[n_computed_pairs] = dh
                    dv_of_pairs[n_computed_pairs] = dv
                    zrec_of_pairs[n_computed_pairs] = z_rec
                    zsrc_of_pairs[n_computed_pairs] = z_src

                    n_computed_pairs += 1
                t2 = perf_counter()

                if ipair % 10000 == 0:
                    ETA = (t2-t1)*(npairs-ipair)/3600.
                    print(f"On {ipair=} of {npairs} {n_computed_pairs=} ({n_computed_pairs/npairs*100}% reduction) {ETA=}h")

                ipair += 1
     

        pairs_to_compute = pairs_to_compute[:n_computed_pairs,:]
        dd_of_pairs = dd_of_pairs[:n_computed_pairs]
        dh_of_pairs = dh_of_pairs[:n_computed_pairs]
        dv_of_pairs = dv_of_pairs[:n_computed_pairs]
        zrec_of_pairs = zrec_of_pairs[:n_computed_pairs]
        zsrc_of_pairs = zsrc_of_pairs[:n_computed_pairs]

        print(f"Need only {n_computed_pairs} pairs of {npairs} ({n_computed_pairs/npairs*100}% reduction)")

        if store_here is not None:
            import h5py
            with h5py.File(store_here + '.h5', 'w') as hf:
                # hf.create_dataset("dists", data=dists)
                hf.create_dataset("pairs_to_compute", data=pairs_to_compute)
                hf.create_dataset("dd_of_pairs", data=dd_of_pairs)
                hf.create_dataset("dh_of_pairs", data=dh_of_pairs)
                hf.create_dataset("dv_of_pairs", data=dv_of_pairs)
                hf.create_dataset("zrec_of_pairs", data=zrec_of_pairs)
                hf.create_dataset("zsrc_of_pairs", data=zsrc_of_pairs)
                hf.create_dataset("delta_h", data=delta_h)
                hf.create_dataset("delta_v_rec", data=delta_v_rec)
                hf.create_dataset("delta_v_src", data=delta_v_src)


        # return dists, pairs_to_compute, dh_of_pairs, dv_of_pairs, zrec_of_pairs, zrec_of_pairs
        return 

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

        # if verbose:
        #     print("_call_core_fast")
        #     # print(f"        psource = {psource}")
        #     print(f"        psource.x = {psource.x}")
        #     # print(f"        station = {station}")
        #     print(f"        station.x = {station.x}")

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
        # print(f"{tdata=}")
        # print(f"{tdata.shape=}")
        tdata_ = tdata.T
        tdata_ = tdata_.reshape((1, tdata_.shape[0], tdata_.shape[1]))
        # tdata_ = tdata
        # tdata_ = tdata_.reshape((1, tdata_.shape[1], tdata_.shape[0]))
        z, e, n, t0 = core.subgreen2(mb, src, rcv, stype, updn, d, a, b, rho, qa, qb, dt, nfft, tb, nx, sigma,
                                           smth, wc1, wc2, pmin, pmax, dk, kc, taper, x, pf, df, lf, tdata_, sx, sy, rx, ry)

        self._logger.debug('ShakerMaker._call_core_fast - core.subgreen returned: z_size'.format(len(z)))

        return z, e, n, t0
