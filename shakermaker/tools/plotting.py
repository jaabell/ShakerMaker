# -*- coding: utf-8 -*-
"""

A set of useful little plotting functions using 
matplotlib. 

"""

import imp
import numpy as np
import matplotlib.pyplot as plt
from shakermaker.station import Station
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


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


def ZENTPlot(station, fig=0, show=False, xlim=[], label=[], integrate=0, differentiate=0, savefigname="", linestyle="-", linewidth=2):
    """Plot (using matplotlib) the response at a given station.

    :param station: The station response to plot.
    :type station: :obj:`shakermaker.Receiver`
    :param fig: Figure number to plot on.
    :type fig: int. 
    :param show: Invoke ``plt.show()``.
    :type show: bool
    :param xlim: Set x-limits to xlim.
    :type xlim: list
    :param label: Invoke ``plt.show()``.
    :type label: string
    :param integrate: Show integral of response (``integrate`` times)
    :type integrate: int
    :param differentiate: Show integral of response (``differentiate`` times)
    :type differentiate: int

    """
    
    if rank == 0: #Only P0 is allowed to plot

        assert isinstance(station, Station), "station must be an instance of the shakermaker.Station class"

        if integrate == 0 and differentiate == 0:
            z,e,n,t = station.get_response()
        elif integrate > 0 and differentiate == 0:
            z,e,n,t = station.get_response()
            # z,e,n,t = station.get_response_integral(ntimes=integrate)
            z = cumulative_trapezoid(z.copy(), t, initial=0.)
            e = cumulative_trapezoid(e.copy(), t, initial=0.)
            n = cumulative_trapezoid(n.copy(), t, initial=0.)
        elif differentiate > 0 and integrate == 0:
            # z,e,n,t = station.get_response_derivative(ntimes=differentiate)
            # z,e,n,t = station.get_response_derivative(ntimes=differentiate)
            z_,e_,n_,t = station.get_response()
            #z = np.gradient(z, t)
            Nt = len(t)
            dt = t[1] - t[0]
            z = np.zeros(Nt)
            z[1:] = (z_[1:] - z_[0:-1])/dt
            #e = np.gradient(e, t)
            e = np.zeros(Nt)
            e[1:] = (e_[1:] - e_[0:-1])/dt
            #n = np.gradient(n, t)
            n = np.zeros(Nt)
            n[1:] = (n_[1:] - n_[0:-1])/dt
        else:
            print(f"Not allowed to pass integrate={integrate} and differentiate={differentiate} simultaneously. ")
            return 0

        if fig == 0:
            fighandle = plt.figure()

        else:
            fighandle = plt.figure(fig)

        for i,comp in enumerate([z,e,n]):
            if i == 0:
                ax0 = plt.subplot(3,1,i+1)
            else:
                plt.subplot(3,1,i+1,sharex=ax0,sharey=ax0)
            plt.plot(t,comp, label=label, linestyle=linestyle, linewidth=linewidth)
            if len(xlim) == 2:
                plt.xlim(xlim)
            if integrate == 0 and differentiate == 0:
                plt.ylabel(["$\\dot{u}_Z$","$\\dot{u}_E$","$\\dot{u}_N$"][i])
            elif integrate > 0 and differentiate == 0:
                plt.ylabel(["$u_Z$","$u_E$","$u_N$"][i])
            elif differentiate > 0 and integrate == 0:
                plt.ylabel(["$\\ddot{u}_Z$","$\\ddot{u}_E$","$\\ddot{u}_N$"][i])
        plt.xlabel("Time, $t$ (s)")
        plt.suptitle(station.metadata["name"])

        if show:
            plt.show()

        if len(savefigname) > 0:
            plt.savefig(savefigname)

        return fighandle

    else:
        return 0

def StationPlot(stations, fig=0, show=False, autoscale=False):
    """Plot (using matplotlib) a set of Receiver stations.

    :param stations: The station to plot. 
    :type station: :obj:`shakermaker.Receiver`
    :param fig: Figure number to plot on.
    :type fig: int. 
    :param autoscale: Try to make the axes equal.
    :type autoscale: bool

    """

    if rank > 0: #Only P0 is allowed to plot
        return 0

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if fig == 0:
        fighandle = plt.figure()
        ax = fighandle.add_subplot(111, projection='3d')
    else:
        fighandle = plt.figure(fig)
        ax = plt.gca()

    n_stations = stations.nstations

    x_rcv = np.zeros(n_stations)
    y_rcv = np.zeros(n_stations)
    z_rcv = np.zeros(n_stations)

    for i, rcv in enumerate(stations):
        x = rcv.x
        x_rcv[i] = x[0]
        y_rcv[i] = x[1]
        z_rcv[i] = -x[2]

    ax.scatter(y_rcv, x_rcv,  z_rcv,  "b")#, c=-z_rcv)


    if autoscale:
        # ax.auto_scale_xyz(y_rcv, x_rcv, z_rcv)
        # ax.autoscale(enable=True, axis='both', tight=None)
        max_range = np.array([x_rcv.max()-x_rcv.min(), y_rcv.max()-y_rcv.min(), z_rcv.max()-z_rcv.min()]).max() / 2.0
        mid_x = (x_rcv.max()+x_rcv.min()) * 0.5
        mid_y = (y_rcv.max()+y_rcv.min()) * 0.5
        mid_z = (z_rcv.max()+z_rcv.min()) * 0.5
        ax.set_ylim(mid_x - max_range, mid_x + max_range)
        ax.set_xlim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.invert_zaxis()

    ax.set_xlabel("(Y) Easting (km)")
    ax.set_ylabel("(X) Northing (km)")
    ax.set_zlabel("(Z) Depth (km)")

    if show:
        plt.show()

    return fighandle


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def SourcePlot(sources, fig=0, show=False, autoscale=False, colorby="maxstf", colorbar=False, axes_equal=True):
    """Plot (using matplotlib) a set of sources.

    :param sources: The sources to plot. 
    :type station: :obj:`shakermaker.SourceModel`
    :param fig: Figure number to plot on.
    :type fig: int. 
    :param colorby: Color sources by
    :type colorby: ``str`` values: ``"maxstf"|"strike"|"dip"|"rake"|tt"``
    :param colorbar: Add a colorbar. 
    :type colorbar: bool
    :param axes_equal: Set 3-D axes to equal scale. 
    :type axes_equal: bool

    ``colorby`` specifications are:

    * ``"maxstf"`` : Maximum value of source time function
    * ``"strike"`` : Strike angle
    * ``"dip"`` :  Dip angle
    * ``"rake"`` :  Rake angle
    * ``"tt"`` : Trigger time

    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.integrate import trapz
    
    if fig == 0:
        fighandle = plt.figure()
        ax = fighandle.add_subplot(111, projection='3d')
    else:
        fighandle = plt.figure(fig)#, facecolor="white")
        ax = plt.gca()
        # ax.set_facecolor("white")

    n_sources = sources.nsources

    x_src = np.zeros(n_sources)
    y_src = np.zeros(n_sources)
    z_src = np.zeros(n_sources)
    c_src = np.zeros(n_sources)

    case = {
        "maxstf" : 0,
        "strike" : 1,
        "dip" : 2,
        "rake" : 3,
        "tt" : 4,
        "slip" : 5
    }[colorby]

    clabel = {
        "maxstf" : "Max of STF",
        "strike" : "Strike",
        "dip" : "Dip",
        "rake" : "Rake",
        "tt" : "Trigger time",
        "slip" : "Max Slip"
    }[colorby]


    for i, src in enumerate(sources):
        x = src.x
        stf = src.stf
        angles = src.angles
        tt = src.tt

        stf.dt = 0.01
        if case==0: #"maxstf"
            c = stf.data.max()
        elif case==1: #"strike"
            c = angles[0]*180/np.pi
        elif case==2: #"dip"
            c = angles[1]*180/np.pi
        elif case==3: #"rake"
            c = angles[2]*180/np.pi
        elif case==4: #"tt"
            c = tt
        elif case==5: #"slip"
            t = stf.t
            v = stf.data
            slip = trapz(v, t)
            c = slip

        x_src[i] = x[0]
        y_src[i] = x[1]
        z_src[i] = -x[2]
        c_src[i] = c

    theplot = ax.scatter(y_src, x_src,  z_src,  c=c_src)


    if autoscale:
        # ax.auto_scale_xyz(y_src, x_src, z_src)
        # ax.autoscale(enable=True, axis='both', tight=None)
        max_range = np.array([x_src.max()-x_src.min(), y_src.max()-y_src.min(), z_src.max()-z_src.min()]).max() / 2.0
        mid_x = (x_src.max()+x_src.min()) * 0.5
        mid_y = (y_src.max()+y_src.min()) * 0.5
        mid_z = (z_src.max()+z_src.min()) * 0.5
        ax.set_ylim(mid_x - max_range, mid_x + max_range)
        ax.set_xlim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if axes_equal:
        set_axes_equal(ax)


    ax.invert_zaxis()

    ax.set_xlabel("(Y) Easting (km)")
    ax.set_ylabel("(X) Northing (km)")
    ax.set_zlabel("(Z) Depth (km)")

    if colorbar:
        cbar = fighandle.colorbar(theplot)
        cbar.set_label(clabel)

    if show:
        plt.show()

    return fighandle


