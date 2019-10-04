      subroutine subrt(mb,stype,src,rcv,
     &  a,b,qa,qb,d,rho,mu,xi,si,
     &  vs,hs,xmax,flip,updn,
     &  nfft,dt,nx,x,t0,fout,tdata,
     &  rf,df,lf,sx,sy,sz,rx,ry,rz,z,e,n,
     &  sigma,taper,nb,smth,wc1,wc2,
     &  pmin,pmax,dk,kmax)

      IMPLICIT NONE

      include 'constants.h'
      include 'model.h'
      logical dynamic
      integer i,j,l,iblank,nfft,nfft2,nfft3,n,ix,nx,tenPerc,count,total
      integer nCom,wc,wc1,wc2,tb,smth,idx0,flip
      real k,omega,dt,dk,dw,sigma,const,phi,hs,xmax,vs
      real dfac,pmin,pmax,kc,taper,filter
      real qa(nlay),qb(nlay),a(nlay),b(nlay),x(ndis),t0(ndis)
      complex w,att,nf,u(3,3)
      real aj0,aj1,aj2,z,tdata(ndis,9,2*nt)
      complex sum(ndis,9,nt), data(nt)
      character*80 fout(ndis)
      real tt0
      real zz(2*nt),ee(2*nt),nn(2*nt)
c
c***************************************************************
c input velocity model
      !open(1,file="green.in")
      !read(1,*)mb,src,stype,rcv,updn
      nCom = 3 + 3*stype
      idx0 = 47
      if (stype.eq.0) idx0 = 96

      flip = 1
      if ( rcv.GT.src ) then    ! flip the model so rcv is above src
         flip = -1
         src = mb - src + 2
         rcv = mb - rcv + 2
      endif

      hs = 0.
      do i = 1, mb
         !read(1,*)d(i),a(i),b(i),rho(i),qa(i),qb(i) *JC this should be
         !passed to the subroutine
         if (b(i).lt.epsilon) b(i) = epsilon
         mu(i) = rho(i)*b(i)*b(i)
         xi(i) = b(i)*b(i)/(a(i)*a(i))
         hs    = hs + d(i)
      enddo
      if (flip.lt.0) then
         d=d(mb:1:-1)
         a=a(mb:1:-1)
         b=b(mb:1:-1)
         qa=qa(mb:1:-1)
         qb=qb(mb:1:-1)
         rho=rho(mb:1:-1)
         mu=mu(mb:1:-1)
         xi=xi(mb:1:-1)
      endif
      vs = b(src)

      call source(stype, xi(src), mu(src), si, flip)

      !read(1,*) sigma,nfft,dt,taper,tb,smth,wc1,wc2
      !read(1,*) pmin,pmax,dk,kc
!2 512 0.2 0.5 25 2 1 1  # sigma nt dt taper nb smth wc1 wc2
!0.  1 0.3 15    # pmin pmax dk kmax

c input distance ranges
      !read(1,*) nx *JC this should be passed to the subroutine
      xmax = hs
      do ix=1,nx
        !write(0, '(a)') 'Input x t0 output_name (2f10.3,1x,a)'
        !read(1,*)x(ix),fout(ix) *JC this should be passed to the
        !subroutine
        call subtrav(mb,a,d,src,rcv,x(ix),tt0)
        !write(*,*) 'TT0 IS =',tt0
        t0(ix)=tt0
        if (xmax .LT. x(ix)) xmax=x(ix)
        t0(ix) = t0(ix)-tb*dt
      enddo

      !close(1) *JC closed  it becuase no longer reading.

      call subfk(mb,stype,src,rcv,a,b,qa,qb,d,rho,mu,xi,si,
     &  vs,hs,xmax,flip,updn,
     &  nfft,dt,nx,x,t0,fout,tdata)

      !call subfocal(0*pi/180.0,90*pi/180.0,180*pi/180.0
     !&   ,tdata,0.0,0.0,0.0,1.,1.,0.,zz,ee,nn)
      call subfocal(pf,df,lf,tdata,sx,sy,sz,rx,ry,rz,zz,ee,nn)
     return
      end
