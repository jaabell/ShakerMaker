      subroutine subtrav(ne,vab,hh,srclay,rcvlay,x,tt0)
      !program travel
c============================================================
c Calculate travel time for horizontal layered model.
c It outputs both times for first arrival and direct arrival.
c Revision history:
c       Nov. 17, 2008   Now it can take receiver at any depth.
c============================================================
      IMPLICIT NONE
      INCLUDE 'aseries.h'
      REAL t, td, x, aa
      COMPLEX taup, pd, p
      INTEGER i, k, nx, rcv_lay
cJORGE
      real :: tt0
      real, intent(in) ::  vab(max_num_lay),hh(max_num_lay)
      integer, intent(in) :: srclay,rcvlay,ne 
      num_lay=ne
      src_lay=srclay
      rcv_lay=rcvlay
      nx=1
cJORGE
      !open(5,file='trav_input.dat')
      !open(6,file='trav_input.dat')
c Note that the source and receiver are located at the bottom of the layer.
      !write(0,*)'Input number_of_layers layer_src_at_botom rcv_layer'
      !open(5,file='input.dat')
      !read(5,*)num_lay,src_lay,stype,rcv_lay,updn
      !write(*,*)mb,src,stype,rcv,updn
      !read(5,*)sigma,nfft,dt,taper,tb,smth,wc1,wc2
      !write(*,*)sigma,nfft,dt,taper,tb,smth,wc1,wc2
      !read(5,*)pmin,pmax,dk,kc
      !write(*,*)pmin,pmax,dk,kc



      !read (5,*) num_lay,src_lay,rcv_lay
      IF (src_lay.LT.rcv_lay) THEN
         i = src_lay
         src_lay = rcv_lay
         rcv_lay = i
      ENDIF
      IF (num_lay .GT. max_num_lay) THEN
         WRITE(0,*) ' Number of layers exceeds max. allowed'
         CALL EXIT(1)
      ENDIF
      !write(0,*)'Input thickness of each layer and velocity'
      do i=1,num_lay
        !read(5,*) thk(i),vv(i),vps(1,i),den(i),qq(i),q(1,i)
        vps(1,i)=vab(i)
        thk(i)=hh(i)
        if (vps(1,i).LT.Vmin) vps(1,i) = Vmin
        vps(1,i) = 1./vps(1,i)**2
        vps(2,i) = vps(1,i)
      enddo

      !write(0,*)'Input the number of distance ranges'
      !open(6,file='dist.dat')
      !open(7,file='range.dat')
      !read(6,*) nx
      !write(0,*)'Input distance range (in km)'
      !write(7,*)nx      
      do i=1,nx
      !  read(6,*) x
c direct arrival
        topp = rcv_lay + 1
        bttm = src_lay
        aa = 1.E+20
        do k = topp, bttm
           ray_len(1,k) = thk(k)
           ray_len(2,k) = 0.
           aa = AMIN1(aa,vps(1,k))
        enddo
        pd = CMPLX(SQRT(aa),1.E-20)
        call findp0(x,pd)
        td = taup(pd,x)
        t0 = td
        p0 = pd
c reflected arrivals from below.
        do bttm = src_lay+1,num_lay-1
           ray_len(1,bttm) = 2.*thk(bttm)
           ray_len(2,bttm) = 0.
           aa = AMIN1(aa,vps(1,bttm))
           p = CMPLX(SQRT(aa),1.E-20)
           call findp0(x,p)
           aa = AMIN1(aa,vps(1,bttm+1))
           pc = CMPLX(SQRT(aa),1.E-20)
           if (REAL(p) .GT. REAL(pc)) p = pc
           t = taup(p,x)
           if (t.lt.t0) then
              t0=t
              p0=p
           endif
        enddo
c reflected arrivals from above.
        bttm = src_lay
        do topp = rcv_lay,1,-1
C          if (topp.EQ.0) exit
           ray_len(1,topp) = 2.*thk(topp)
           ray_len(2,topp) = 0.
           aa = AMIN1(aa,vps(1,topp))
           p = CMPLX(SQRT(aa),1.E-20)
           call findp0(x,p)
           if (topp.GT.1) aa = AMIN1(aa,vps(1,topp-1))
           pc = CMPLX(SQRT(aa),1.E-20)
           if (REAL(p) .GT. REAL(pc)) p = pc
           t = taup(p,x)
           if (t.lt.t0) then
              t0=t
              p0=p
           endif
        enddo
c output
        !write(7,'(f8.2,f8.2,f8.2,f8.4,f8.4)')x,t0,td,REAL(p0),REAL(pd)
        !write(7,*)x,t0,x
      enddo
      !close(5)
      !close(6)
      !close(7)
      tt0=t0
      return
      !stop
      end subroutine 

