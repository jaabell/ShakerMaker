      program fk
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
      real aj0,aj1,aj2,z,tdata(2*nt)
      complex sum(ndis,9,nt), data(nt)
      character*80 fout(ndis)

      dynamic = .TRUE.

c***************************************************************
c input velocity model
      open(1,file="green.in")
      write(0,'(a)') 'Input nlay src_layer src_type rcv_layer updn'
      read(1,*)mb,src,stype,rcv,updn
      if (mb.gt.nlay .OR. src.EQ.rcv) then
         write(0,*) mb,'>',nlay,' or source receiver at same level'
         call exit(1)
      endif
      nCom = 3 + 3*stype
      idx0 = 47
      if (stype.EQ.0) idx0 = 96

      flip = 1
      if ( rcv.GT.src ) then    ! flip the model so rcv is above src
         flip = -1
         src = mb - src + 2
         rcv = mb - rcv + 2
      endif

      hs = 0.
      do i = 1, mb
         write(0,*) 'Input thickness Vp Vs rho Qa Qb for layer',i
         j = i
         if ( flip.LT.0 ) j = mb-i+1
         read(1,*)d(j),a(j),b(j),rho(j),qa(j),qb(j)
         if (b(j).LT.epsilon) b(j) = epsilon
         mu(j) = rho(j)*b(j)*b(j)
         xi(j) = b(j)*b(j)/(a(j)*a(j))
         write(0,'(i4,4f7.2,2e9.2)')i,d(j),a(j),b(j),rho(j),qa(j),qb(j)
         if ( j.LT.src .AND. j.GE.rcv ) hs = hs + d(j)
      enddo
      write(0,'(a15,f8.3)') 'source-station separation =',hs

      vs = b(src)
      call source(stype, xi(src), mu(src), si, flip)

c input sigma, number of points, dt, taper, and samples before first-arr
c sigma is the small imaginary part of the complex frequency, input in
c 1/time_length. Its role is to damp the trace (at rate of exp(-sigma*t))
c to reduce the wrape-arround.
      write(0,'(a)') 'Input sigma NFFT dt lp nb smooth_factor wc1 wc2'
      read(1,*) sigma,nfft,dt,taper,tb,smth,wc1,wc2
      if ( nfft.EQ.1 ) then
         dynamic = .FALSE.
         nfft2 = 1
         wc1 = 1
      else
         nfft2 = nfft/2
      endif
      if (nfft2*smth .GT. nt) then
         write(0,'(a)') 'Number of points exceeds allowed'
         call exit(1)
      endif
      dw = pi2/(nfft*dt)
      sigma = sigma*dw/pi2
      wc = nfft2*(1.-taper)
      if (wc .LT. 1) wc=1
      taper = pi/(nfft2-wc+1)
      if (wc2.GT.wc) wc2=wc
      if (wc1.GT.wc2) wc1=wc2

c Input phase velocity window, dk, and kmax at w=0.
c pmin and pmax are in 1/vs.
c dk is in pi/max(x,hs). Since J(kx) oscillates with period 2pi/x at
c large distance, we require dk < 0.5 to guarentee at least 4 samples per
c period.
c kmax is in 1/hs. Because the kernels decay with k at rate of exp(-k*hs) at
c w=0, we require kmax > 10 to make sure we have have summed enough.
      write(0,'(a)') 'Input pmin pmax dk kc'
      read(1,*) pmin,pmax,dk,kc
      kc = kc/hs
      pmin = pmin/vs
      pmax = pmax/vs

c input distance ranges
      write(0,'(a)') 'Input number of distance ranges to compute'
      read(1,*) nx
      if (nx.gt.ndis) then
         write(0,'(a)') 'Number of ranges exceeds the max. allowed'
         call exit(1)
      endif
      xmax = hs
      do ix=1,nx
        write(0, '(a)') 'Input x t0 output_name (2f10.3,1x,a)'
        read(1,'(2f10.3,1x,a)')x(ix),t0(ix),fout(ix)
        if (xmax .LT. x(ix)) xmax=x(ix)
        t0(ix) = t0(ix)-tb*dt
      enddo
      dk = dk*pi/xmax
      const = dk/pi2

      close(1)
c***************************************************************
c*************** do wavenumber integration for each frequency
      z = pmax*nfft2*dw/kc
      k = sqrt(z*z+1)
      total = nfft2*(kc/dk)*0.5*(k+log(k+z)/z)
c     write(0,'(a3,f9.5,a8,f9.2,a8,i9)')'dk',dk,'kmax',kc,'N',total
      write(0,1001)'dk =',dk,'kmax =',kc,'pmax =',pmax,'N =',total
1001  format(a6,f9.5,a8,f6.2,a8,f6.4,a8,i9)
      tenPerc = 0.1*total
      count = 0.
      kc = kc*kc
      do j=1,nfft2
         do ix = 1,nx
            do l =1,nCom
               sum(ix,l,j) = 0.
            enddo
         enddo
      enddo
      write(0,*)' start F-K computation, iw-range:',wc1,wc2,wc,nfft2
      do j=wc1, nfft2           ! start frequency loop
         omega = (j-1)*dw
         w = cmplx(omega,-sigma)        ! complex frequency
         do i = 1, mb
            att = clog(w/pi2)/pi + cmplx(0.,0.5)                ! A&R, p182
            ka(i) = w/(a(i)*(1.+att/qa(i)))
            kb(i) = w/(b(i)*(1.+att/qb(i)))
            ka(i) = ka(i)*ka(i)
            kb(i) = kb(i)*kb(i)
         enddo
         k = omega*pmin + 0.5*dk
         n = (sqrt(kc+(pmax*omega)**2)-k)/dk
         do i=1,n               ! start k-loop
            call kernel(k,u,mb,stype,src,rcv,updn,ka,kb,d,rho,mu,xi,si)
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
            count=count+1
            if ( mod(count,tenPerc) .EQ. 0) then
               write(0,'(i4,a6)') int(100.*count/total)+1, '% done'
            endif
         enddo                  ! end of k-loop
         filter = const
         if ( dynamic .AND. j.GT.wc )
     +          filter = 0.5*(1.+cos((j-wc)*taper))*filter
         if ( dynamic .AND. j.LT.wc2 )
     +          filter = 0.5*(1.+cos((wc2-j)*pi/(wc2-wc1)))*filter
         do ix=1,nx
            phi = omega*t0(ix)
            att = filter*cmplx(cos(phi),sin(phi))
            do l=1,nCom
               sum(ix,l,j) = sum(ix,l,j)*att
            enddo
         enddo
      enddo                     ! end of freqency loop
      write(0,'(i9,a40)') count,' 100% done, writing files ... '
      
c***************************************************************
c*************** do inverse fourier transform
      dt = dt/smth
      nfft = smth*nfft
      nfft3 = nfft/2
      dfac = exp(sigma*dt)
      do ix=1,nx
         if ( nfft2.EQ.1 ) then
            write(*,'(f5.1,9e11.3)')x(ix),(real(sum(ix,l,1)),l=1,nCom)
         else
            iblank = index(fout(ix),' ')
            fout(ix)(iblank+1:iblank+1) = char(0)
            do l=1,nCom
               do j=1,nfft2
                  data(j) = sum(ix,l,j)
               enddo
               do j=nfft2+1,nfft3
                  data(j) = 0.
               enddo
               call fftr(data,nfft3,-dt)
               z = exp(sigma*t0(ix)) ! removing damping due sigma. Damping is w.r.t t=0
               do j=1,nfft3
                  tdata(2*j-1) = real(data(j))*z
                  z = z*dfac
                  tdata(2*j) = aimag(data(j))*z
                  z = z*dfac
               enddo
               fout(ix)(iblank:iblank) = char(idx0+l)
               call wrtsac0(fout(ix),dt,nfft,t0(ix),x(ix),tdata)
            enddo
         endif
      enddo
      
      stop
      end
