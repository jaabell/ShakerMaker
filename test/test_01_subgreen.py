###############################################################################
# shakermaker Test Suite
# Test # 01 - Subgreen
# file: /tests/test_01_subgreen.py
#
# Description
#
# This is a direct call to the subgreen fortran function, interfaced through
# f2py. 
#
###############################################################################

import matplotlib.pyplot as plt
import scipy as sp
import shakermaker
import shakermaker.core

# exit(0)

mb,src,stype,rcv,updn = 11,5,2,1,0
a = [5.21,5.37,5.55,5.72,5.89,5.98,6.80,7.01,7.55,8.05,8.05]
b = [2.99,3.09,3.19,3.29,3.39,3.44,3.81,3.95,4.24,4.39,4.39]
rho = [2.5,2.5,2.6,2.7,2.7,2.8,2.8,2.9,3.0,3.4,3.4]
d = [2.5,2.0,2.0,2.0,2.0,4.5,10.0,15.0,10.0,20.0,0.0]
qa = [300,300,300,300,300,300,600,600,600,600,600]
qb = [150,150,150,150,150,150,300,300,300,300,300]
  
  
sigma=2
nfft=4096
dt=0.05
taper=0.5
tb=1000
smth=2
wc1=1
wc2=1

pmin=0.
pmax=1
dk=0.3
kc=15

nx=1

x = 50.
tdata = sp.zeros((500,9,16384))
sx,sy,sz,rx,ry,rz = 0.,0.,0.,1.,1.,1.
pf,df,lf = 0.,0.,0.
tdata,z,e,n,t0 = shakermaker.core.subgreen(mb,src,rcv,stype,updn,d,a,b,rho,qa,qb,dt,nfft,tb,nx,sigma,smth,wc1,wc2,pmin,pmax,dk,kc,taper,x,pf,df,lf,sx,sy,rx,ry)

# Parameters
# ----------
# mb : input int
# src : input int
# rcv : input int
# stype : input int
# updn : input int
# d : input rank-1 array('f') with bounds (2000)
# a : input rank-1 array('f') with bounds (2000)
# b : input rank-1 array('f') with bounds (2000)
# rho : input rank-1 array('f') with bounds (2000)
# qa : input rank-1 array('f') with bounds (2000)
# qb : input rank-1 array('f') with bounds (2000)
# dt : input float
# taper : input float
# tb : input int
# nx : input int
# x : input rank-1 array('f') with bounds (500)
# pf : input float
# df : input float
# lf : input float
# tdata : input rank-3 array('f') with bounds (500,9,16384)
# sx : input float
# sy : input float
# sz : input float
# rx : input float
# ry : input float
# rz : input float

# Returns
# -------
# zz : rank-1 array('f') with bounds (16384)
# ee : rank-1 array('f') with bounds (16384)
# nn : rank-1 array('f') with bounds (16384)

Nt = z.shape[0]
tmax = Nt*dt
t = sp.arange(0, tmax, dt)

plt.subplot(3,1,1)
plt.plot(t,z,'k')
plt.subplot(3,1,2)
plt.plot(t,e,'k')
plt.subplot(3,1,3)
plt.plot(t,n,'k')

plt.show()