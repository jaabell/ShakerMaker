      subroutine subfocal(pf,df,lf,tdata,j,nx,nfft,
     & sx,sy,rx,ry,z,e,n)
      IMPLICIT NONE
C       include 'constants.h'
C       include 'model.h'
      integer j,nfft,nx
      real tdata(nx,9,2*nfft)
      real sx,sy,rx,ry
      real rr,dx,dy
      real z(2*nfft),r(2*nfft),t(2*nfft)
      real e(2*nfft),n(2*nfft)

      real f1,f2,f3,n1,n2,n3,lf,pf,df,p,theta

      dx=rx-sx
      dy=ry-sy
      rr=sqrt(dx*dx+dy*dy)
      p=atan2(dy,dx)
      theta=p

      f1 = cos(lf)*cos(pf)+sin(lf)*cos(df)*sin(pf)
      f2 = cos(lf)*sin(pf)-sin(lf)*cos(df)*cos(pf)
      f3 = -sin(lf)*sin(df)
      n1 = -sin(pf)*sin(df)
      n2 = cos(pf)*sin(df)
      n3 = -cos(df)

      z = tdata(j,7,:)*((f1*n1-f2*n2)*cos(2*p)+(f1*n2+f2*n1)*sin(2*p))
     & + tdata(j,4,:)*((f1*n3+f3*n1)*cos(p)+(f2*n3+f3*n2)*sin(p))
     & + tdata(j,1,:)*(f3*n3)

      r = tdata(j,8,:)*((f1*n1-f2*n2)*cos(2*p)+(f1*n2+f2*n1)*sin(2*p))
     & + tdata(j,5,:)*((f1*n3+f3*n1)*cos(p)+(f2*n3+f3*n2)*sin(p))
     & + tdata(j,2,:)*(f3*n3)

      t = tdata(j,9,:)*((f1*n1-f2*n2)*sin(2*p)-(f1*n2+f2*n1)*cos(2*p))
     & + tdata(j,6,:)*((f1*n3+f3*n1)*sin(p)-(f2*n3+f3*n2)*cos(p))

      !later converte to zen cartesian coordinate system.
      e = -r*sin(theta) - t*cos(theta)
      n = -r*cos(theta) + t*sin(theta)

      return

      end
