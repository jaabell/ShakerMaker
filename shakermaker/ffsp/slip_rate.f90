!======================================================================
! The subroutines in this file are for generating the slip rate
! functions for each point source, and sum them together to get
! spectra of larget event
!
! Written by Pengcheng Liu
! Copyright (c) 2005 by Pengcheng Liu
!
!  Simply using the modified yoffe function as default
!         Chen Ji, 2020
!======================================================================
subroutine sum_point_svf(svf)
 use time_freq
 use sp_sub_f
 implicit NONE
 integer:: i,j,j0,j1,j2,jmm
 real:: rise_time
! Windows change: svf_s and mr changed from stack arrays to heap (allocatable).
! Stack arrays of size ntime (131072 reals = 512 KB each) cause stack overflow on Windows.
 real, dimension(ntime):: svf
 real, allocatable:: svf_s(:), mr(:)
  allocate(svf_s(ntime), mr(2*ntime))
  svf=0.0
  mr=0.0
 do i=1,nsum
   rise_time=rstm(i)+pktm(i)
   cft1=pktm(i)/rise_time
! Windows change: pass ntime as nsvf so svf_yoffe uses explicit-size interface.
   call svf_yoffe(ntime,dt,rise_time,cft1,svf_s,ntime)
   j0=int(rptm(i)/dt+0.5)
   j1=j0+1
   j2=int(rise_time/dt)+j1
   do j=j1,j2
! Windows change: bounds guards prevent out-of-bounds write that crashes
! silently on Windows/ifx (Linux/gfortran tolerates it due to memory layout).
     if(j.ge.1 .and. j.le.2*ntime .and. &
        (j-j0).ge.1 .and. (j-j0).le.ntime) then
       mr(j)=mr(j)+slip(i)*svf_s(j-j0)
     endif
   enddo
 enddo
 do i=1,ntime
    svf(i)=mr(i)
 enddo
! Windows change: deallocate heap arrays allocated above.
 deallocate(svf_s, mr)
end subroutine sum_point_svf
!
!======================================================================
!
subroutine peak_slip_rate(pksr)
 use time_freq
 use sp_sub_f
 implicit NONE
 real, dimension(nsum):: pksr
 integer:: i
!
! just for modified yoffe function
!         Chen Ji, 2020
 do i=1,nsum
   pksr(i)=slip(i)/sqrt(pktm(i))/sqrt(rstm(i))
 end do

end subroutine peak_slip_rate
!
!======================================================================
! modified normalized yoffe function. Chen Ji, 2020
!
! Windows change: added nsvf argument so svf uses explicit-size declaration svf(nsvf).
! Passing an assumed-shape (dimension(:)) or fixed-size array from an allocatable
! caller without an explicit interface is illegal Fortran 90 and causes heap
! corruption on Windows/ifx (crash inside allocate). Explicit-size passes by
! address only — no array descriptor, no interface needed, legal on all compilers.
! yoffe and hsin also moved to heap (allocatable) to avoid 3x512 KB on the stack.
subroutine svf_yoffe(nt,dt,rise_time,cft1,svf,nsvf)
 implicit NONE
 real, parameter:: pi=3.14159265
 integer:: nt,nsvf,npt_yoffe,nsin,i,j,nall
 real:: cft1,dt,rise_time
 real:: ty,tsin,sn,t,tsin_min
! Windows change: svf explicit-size (no descriptor), yoffe/hsin heap-allocated.
 real,intent(out):: svf(nsvf)
 real,allocatable:: yoffe(:),hsin(:)
 allocate(yoffe(nt),hsin(nt))
 yoffe=0.0
 hsin=0.0

 svf=0.0
 if(cft1.ge.1.0)then
    write(*,*)"unphysical rise-time function"
    stop
 endif
 tsin_min=2.0*dt
 tsin=cft1*rise_time
 ty=rise_time-Tsin
 npt_yoffe=int(ty/dt+1)+1
 nsin=int(tsin/dt+0.1)+1
 nall=npt_yoffe+nsin
 do i=1,npt_yoffe
    t=i*dt
    if(t.le.ty)then
       yoffe(i)=sqrt(ty-t)/sqrt(t)
    else
       yoffe(i)=0.0
    endif
 enddo
 if(tsin.ge.tsin_min)then
   do i=1,nsin
     t=(i-1)*dt
     if(t.le.tsin)then
       hsin(i)=sin(t*pi/tsin)
     else
       hsin(i)=0.0
     endif
   enddo
   do i=1,npt_yoffe
     do j=1,nsin
! Windows change: guard prevents write past end of svf(nsvf).
       if((i+j-1).le.nsvf) svf(i+j-1)=svf(i+j-1)+yoffe(i)*hsin(j)
     enddo
   enddo
 else
   do i=1,npt_yoffe
     svf(i)=yoffe(i)
   enddo
 endif
 sn=0.0
 do i=1,nall
    sn=sn+svf(i)
 enddo
! Windows change: guard against division by zero if sn=0 after clamping.
 if(sn.ne.0.0) svf=svf/(dt*sn)
! Windows change: deallocate heap arrays.
 deallocate(yoffe,hsin)
end subroutine svf_yoffe
!
!======================================================================
! modified normalized kostrov function at r/v=10
! t0 and t02
subroutine svf_kostrov(nt,dt,rise_time,cft1,svf)
 implicit NONE
 real, parameter:: pi=3.14159265,t0=10.0,t02=t0*t0
 integer:: nt,npt_kostrov,nsin,i,j
 real:: cft1,dt,rise_time
 real:: tk,tsin,sn,t
 real,dimension(nt):: svf,yk,hsin

 svf=0.0
 if(cft1.gt.1.0)then
    write(*,*)"unphysical rise-time function"
    stop
 endif

 tsin=cft1*rise_time
 tk=rise_time-Tsin
 npt_kostrov=int(tk/dt+1.0)+1
 nsin=int(tsin/dt+1.0)+1

 yk(1)=0.0
 do i=2,npt_kostrov
    t=(i-1)*dt
    if(t.le.tk)then
       yk(i)=(t+t0)/sqrt((t+t0)**2.0 -t02)
    else
       yk(i)=0.0
    endif
 enddo

 do i=1,nsin
    t=(i-1)*dt
    if(t.le.tsin)then
       hsin(i)=sin((i-1)*dt*pi/tsin)
    else
       hsin(i)=0.0
    endif
 enddo
 do i=1,npt_kostrov
    do j=1,nsin
        svf(i+j-1)=svf(i+j-1)+yk(i)*hsin(j)
    enddo
 enddo
 do i=1,npt_kostrov+nsin-1
    svf(i)=svf(i+1)
 enddo
 sn=0.0
 do i=1,npt_kostrov+nsin-1
    sn=sn+svf(i)
 enddo
 svf=svf/(dt*sn)
end subroutine svf_kostrov
