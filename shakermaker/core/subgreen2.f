      subroutine subgreen2(mb,src,rcv,stype,updn,d,a,b,rho,qa,qb,
     & dt,nfft,tb,nx,sigma,smth,wc1,wc2,pmin,pmax,dk,
     & kc,taper,x,pf,df,lf,tdata,sx,sy,rx,ry,
     & zz,ee,nn,t0)
      IMPLICIT NONE
      include 'constants.h'
      include 'model.h'
      integer i,j,nfft,ix,nx
      integer nCom,wc1,wc2,tb,smth,idx0,flip
      real dt,dk,sigma,hs,xmax,vs,taper
      real pmin,pmax,kc
      real qa(mb),qb(mb),a(mb),b(mb),x(nx),t0(nx)
      real tdata(nx,9,2*nfft)
      real tt0
      real zz(2*nfft),ee(2*nfft),nn(2*nfft)
      real pf,df,lf,sx,sy,rx,ry

c
c***************************************************************
c input velocity model
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
         if (b(i).lt.epsilon) b(i) = epsilon
         mu(i) = rho(i)*b(i)*b(i)
         xi(i) = b(i)*b(i)/(a(i)*a(i))
         hs    = hs + d(i)
      enddo
      if (flip.lt.0) then
C        Previous form, kept for reference:
C        d=d(mb:1:-1)
C        a=a(mb:1:-1)
C        b=b(mb:1:-1)
C        qa=qa(mb:1:-1)
C        qb=qb(mb:1:-1)
C        rho=rho(mb:1:-1)
C        mu=mu(mb:1:-1)
C        xi=xi(mb:1:-1)
C        Reverse only the first mb layers. d, rho, mu and xi are
C        declared with nlay elements in model.h, so the whole-array
C        assignment wrote past the end of the model (segfault when the
C        receiver is below the source).
         d(1:mb)=d(mb:1:-1)
         a(1:mb)=a(mb:1:-1)
         b(1:mb)=b(mb:1:-1)
         qa(1:mb)=qa(mb:1:-1)
         qb(1:mb)=qb(mb:1:-1)
         rho(1:mb)=rho(mb:1:-1)
         mu(1:mb)=mu(mb:1:-1)
         xi(1:mb)=xi(mb:1:-1)
      endif
      vs = b(src)

C       write(*,*)"source"

      call source(stype, xi(src), mu(src), si, flip)

C       sigma = 2
C       smth = 1
C       wc1 = 1
C       wc2 = 1
C       pmin = 0.0
C       pmax = 1.0
C       dk = 0.1
C       kc = 15.0

c input distance ranges
C       write(*,*)"dist"
      xmax = hs
      do ix=1,nx
        call subtrav(mb,a,d,src,rcv,x(ix),tt0)
        t0(ix)=tt0
        if (xmax .LT. x(ix)) xmax=x(ix)
C         if (xmax .LT. x(ix)) xmax=hs
C         if (xmax .LT. x(ix)) xmax=x(ix)+hs ! OJO!!!
        t0(ix) = t0(ix)-tb*dt
      enddo
C       write(*,*) "xmax=", xmax
C       call subfk(mb,stype,src,rcv,a,b,qa,qb,d,rho,mu,xi,si,
C      &  vs,hs,xmax,flip,updn,
C      &  nfft,dt,smth,sigma,pmin,pmax,dk,kc,taper,nx,x,t0,tdata)
C       write(*,*)"pre-subfocal"
C       do i=1,2*nt
C         write(*,*) i, tdata(j,1,i)
C       enddo
C       write(*,*)"subfocal"
      j = 1
C       call subfocal(pf,df,lf,tdata,j,sx,sy,sz,rx,ry,rz,zz,ee,nn)
      call subfocal(pf,df,lf,tdata,j,nx,nfft,
     &  sx,sy,rx,ry,zz,ee,nn)
C       write(*,*)"donesies"
      return

      end
