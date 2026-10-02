      subroutine subfk(mb,stype,src,rcv,
     &  a,b,qa,qb,d,rho,mu,xi,si,
     &  vs,hs,xmax,flip,updn,
     &  nfft,dt,smth,sigma,pmin,pmax,dk,kc,taper,nx,x,t0,tdata)
      implicit none
      include 'constants.h'
      include 'model.h'
      integer i,j,l,nfft,nfft2,nfft3,n,ix,nx
      integer nCom,wc,smth,idx0,flip
      real k,omega,dt,dk,dw,sigma,const,phi,hs,xmax,vs
      real dfac,pmin,pmax,kc,taper,filter
      real x(nx),t0(nx)
      real qa(mb),qb(mb),a(mb),b(mb)
      complex w,att,nf,u(3,3)
      real aj0,aj1,aj2,z,tdata(nx,9,2*nfft)
      complex sum(nx,9,2*nfft), data(2*nfft)
      complex ka_local(mb), kb_local(mb) 

      nCom = 3 + 3*stype
      idx0 = 47
C       write(*,*) "mb=", mb
C       write(*,*) "stype=", stype
C       write(*,*) "src=", src
C       write(*,*) "rcv=", rcv
C       write(*,*) "a=", a(1:mb)
C       write(*,*) "b=", b(1:mb)
C       write(*,*) "qa=", qa(1:mb)
C       write(*,*) "qb=", qb(1:mb)
C       write(*,*) "d=", d(1:mb)
C       write(*,*) "rho=", rho(1:mb)
C       write(*,*) "mu=", mu(1:mb)
C       write(*,*) "xi=", xi(1:mb)
C       write(*,*) "si=", si
C       write(*,*) "vs=", vs
C       write(*,*) "hs=", hs
C       write(*,*) "xmax=", xmax
C       write(*,*) "flip=", flip
C       write(*,*) "updn=", updn
C       write(*,*) "nfft=", nfft
C       write(*,*) "dt=", dt
C       write(*,*) "nx=", nx
C       write(*,*) "x=", x(1)
C       write(*,*) "t0=", t0(1)
      !if ((mb.gt.nlay).or.(src.eq.rcv).or.(src.eq.1)) then
      !   write(0,*) 'Check source receiver positions'
      !   call exit(1)
      !endif

C       smth = 2
C       sigma=2
C       pmin=0.0
C       pmax=1.11
C       dk=0.1
C       kc=15
C       taper = 0.5

      nfft2 = nfft/2
      dw = pi2/(nfft*dt)
      sigma = sigma*dw/pi2
      wc = int(nfft2*(1.-taper))
      if (wc .LT. 1) wc=1
      taper = pi/(nfft2-wc+1)

      vs=b(src)
      pmin = pmin/vs
      pmax = pmax/vs
      kc = kc/hs
      dk = dk*pi/xmax
      const = dk/pi2

c***************************************************************
c*************** do wavenumber integration for each frequency
      z = pmax*real(nfft2)*dw/kc
      k = sqrt(z*z+1)
!      write(0,1001)'dk =',dk,'kmax =',kc,'pmax =',pmax
!1001  format(a6,f9.5,a8,f6.2,a8,f6.4,a8,i9)
      kc = kc*kc
      do j=1,nfft2
         do ix = 1,nx
            do l =1,nCom
               sum(ix,l,j) = 0.
            enddo
         enddo
      enddo
!$OMP PARALLEL DO DEFAULT(SHARED)
!$OMP& PRIVATE(j,omega,w,att,i,k,n,ix,z,aj0,aj1,aj2,u,nf,l,filter,phi,ka_local,kb_local)
!$OMP& SCHEDULE(DYNAMIC)
      do j=1,nfft2              ! start frequency loop
         omega = (j-1)*dw
         w = cmplx(omega,-sigma)        ! complex frequency
         do i = 1, mb
            att = clog(w/pi2)/pi + cmplx(0.,0.5)                ! A&R, p182
C             ka(i) = w/(a(i)*(1.+att/qa(i)))
C             kb(i) = w/(b(i)*(1.+att/qb(i)))
C             ka(i) = ka(i)*ka(i)
C             kb(i) = kb(i)*kb(i)
            ka_local(i) = w/(a(i)*(1.+att/qa(i)))
            kb_local(i) = w/(b(i)*(1.+att/qb(i))) 
            ka_local(i) = ka_local(i)*ka_local(i)
            kb_local(i) = kb_local(i)*kb_local(i)
         enddo
C          !k = omega*pmin + 0.5*dk
         k=dk/2
         n = int((sqrt(kc+(pmax*omega)**2)-k)/dk)
         do i=1,n               ! start k-loop
            call kernel(k,u,mb,stype,src,rcv,updn,ka_local,kb_local,d,rho,mu,xi,si)
            do ix=1,nx
               z = k*x(ix)
               call besselFn(z,aj0,aj1,aj2)
c n=0
               sum(ix,1,j) = sum(ix,1,j) + u(1,1)*aj0*flip
               sum(ix,2,j) = sum(ix,2,j) - u(1,2)*aj1
               sum(ix,3,j) = sum(ix,3,j) - u(1,3)*aj1
c n=1
               nf =    (u(2,2)+u(2,3))*aj1/z
               sum(ix,4,j) = sum(ix,4,j) + u(2,1)*aj1*flip
               sum(ix,5,j) = sum(ix,5,j) + u(2,2)*aj0 - nf
               sum(ix,6,j) = sum(ix,6,j) + u(2,3)*aj0 - nf
c n=2
               nf = 2.*(u(3,2)+u(3,3))*aj2/z
               sum(ix,7,j) = sum(ix,7,j) + u(3,1)*aj2*flip
               sum(ix,8,j) = sum(ix,8,j) + u(3,2)*aj1 - nf
               sum(ix,9,j) = sum(ix,9,j) + u(3,3)*aj1 - nf
            enddo
            k = k+dk
         enddo                  ! end of k-loop
         filter = const
         if (j.gt.wc) filter = 0.5*(1.+cos((j-wc)*taper))*filter
         do ix=1,nx
            phi = omega*t0(ix)
            att = filter*cmplx(cos(phi),sin(phi))
            do l=1,nCom
               sum(ix,l,j) = sum(ix,l,j)*att
            enddo
         enddo
      enddo                     ! end of freqency loop
!$OMP END PARALLEL DO
c***************************************************************
c*************** extraer sum
C       do ix=1,nx
C          do l=1,nCom
C             do j=1,nfft2
C                spectrum(ix,l,j) = sum(ix,l,j)
C             enddo
C          enddo
C       enddo
c***************************************************************
c*************** do inverse fourier transform
      dt = dt/smth
      nfft = smth*nfft
      nfft3 = nfft/2
      dfac = exp(sigma*dt)
C       write(*,*) "subfk 4"
!$OMP PARALLEL DO DEFAULT(SHARED) 
!$OMP& PRIVATE(ix,l,j,data,z)
!$OMP& SCHEDULE(STATIC)
      do ix=1,nx
         !if ( nfft2.EQ.1 ) then
         !   write(*,'(f5.1,9e11.3)')x(ix),(real(sum(ix,l,1)),l=1,nCom)
         !else
            !iblank = index(fout(ix),' ')
            !fout(ix)(iblank+1:iblank+1) = char(0)
            do l=1,nCom
               do j=1,nfft2
                  data(j) = sum(ix,l,j)
               enddo
               do j=nfft2+1,nfft3
                  data(j) = 0.
               enddo
               call fftr(data,nfft3,-dt)
               z = exp(sigma*t0(ix)) ! removing damping due sigma. Damping is w.r.t t=0
C                write(*,*) "l=", l, "/(1)"
               do j=1,nfft3
                  tdata(ix,l,2*j-1) = real(data(j))*z
                  z = z*dfac
                  tdata(ix,l,2*j) = aimag(data(j))*z
                  z = z*dfac
               enddo
C                write(*,*) "l=", l, "(2)"
               !fout(ix)(iblank:iblank) = char(idx0+l)
               !call wrtsac0(fout(ix),dt,nfft,t0(ix),x(ix),tdata(ix,l,:))
            enddo
         !endif
      enddo
!$OMP END PARALLEL DO
      nfft=nfft/smth
      return 
      end