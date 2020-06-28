% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
%    msqy=msqy_quadrant13(m,S)
% computes the integral of x(2)^2*normpdf(x,m,S) across the entire 1st and 3rd quadrants
% m: mean (vector of length 2)
% S: covariance matrix (size [2,2]
%
function msqy=msqy_quadrant13(m,S)

if nargin<1,
   test_msqy_quadrant13();
   msqy=NaN;
   return;
end;
m=m(:);
if length(m)~=2,
   error('invalid dimension of m');
end;
if ~all(size(S)==2),
   error('invalid dimension of S');
end;
   
Ny=100; % number of sections along the longer main axis

[V,L]=eig(S);
L=diag(L);
[L,i]=sort(L);
V=V(:,i);    % short and long main axis vectors are in the first and second columns of V
if det(V)<0,
   V(:,1)=-V(:,1);
end;
if V(1,1)<0,   % asure that short main axis point towards positive x(1) direction
   V=-V;
end;


msqy=0;
% Main axis transformation: y=V'*(x-m) ; x=V*y+m
% inequality equations of the first quadrant are x'*[1, 0;
%                                                   [0, 1]  >0
% transform into V(1,1)*y(1)+V(1,2)*y(2)+m(1)>0
%                V(2,1)*y(1)+V(2,2)*y(2)+m(2)>0
%


sd_y2=L(2)^0.5;
if L(1)<1e-8,
   if L(1)^2<1e-8,
      % we return m(2)^2 im m is in the 1st or 3rd quadrant
      if all(m)>0 || all(m)<0,
         msqy=m(2)^2;
      end;
   else
      % in this case, we just have to decide to which quadrant the point on the long main axis at [0;y(k)] belongs
      % The limits for the entire gaussian in y(2):  +-5*L(2)
      % The integrate along lines with constant y(2) results in normpdf(y(k),0,sd_y)
      ylims=6*[-1,1]*sd_y2;
      y=ylims(1)+diff(ylims)*(0:Ny-1)/(Ny-1);
      dy=diff(y(1:2));
      for k=1:Ny,
         xk=V*[0;y(k)]+m;
         if all(xk>0) || all(xk)<0,
            msqy=msqy+dy*xk(2)^2*normpdf(y(k),0,sd_y2);
         end;
      end;
   end;
else
   % we integrate along lines with constant x(2)
   
   xlims2=m(2)+6*[-1,1]*S(2,2);
   x2_c=xlims2(1)+(0:Ny-1)'*diff(xlims2)/(Ny-1);
   dx2=diff(x2_c(1:2));
   v_m=S(1,1)-S(1,2)/S(2,2)*S(2,1);
   sd_m=v_m^0.5;
   for k=1:Ny,
      % conditional distribution for y(k)
      %  1/2/pi/D^0.5*exp(-0.5*([x(k);x2_c]-m)'*SI*([x(k);x2_c]-m))/P_margin
      %  = 1/2/pi/v_m^0.5*exp(-0.5*(x(k)-m_m)^2/v_m)
      %     with m_m=m(1)+S(1,2)/S(2,2)*(x2_c-m(2))
      %          v_m=S(1,1)-S(1,2)/S(2,2)*S(2,1)
      %
      %     P_margin=1/(2*pi*S(2,2))^0.5*exp(-0.5*(x2_c-m(2))^2/S(2,2))
      
      m_m=m(1)+S(1,2)/S(2,2)*(x2_c(k)-m(2));
      P_margin=1/(2*pi*S(2,2))^0.5*exp(-0.5*(x2_c(k)-m(2))^2/S(2,2));
      if x2_c(k)<0  % 3rd quadrant
         msqy=msqy+x2_c(k)^2*normcdf(0,m_m,sd_m)*P_margin*dx2;
      elseif x2_c(k)>0, % 1st quadrant
         msqy=msqy+x2_c(k)^2*(1-normcdf(0,m_m,sd_m))*P_margin*dx2;
      end;
         
   end;
end;
end

function test_msqy_quadrant13()
rng(762345);
S=randn(2,2);
S=S+S';
[V,L]=eig(S);
S=V'*diag([2,1])*V;

m=[50;50];

figure(1);
clf
hold on 
[h,slope,m]=plot_covariance_ellipse(m,S,0.05);

tic
z2=msqy_quadrant13(m,S);
toc

tic
if false,  % check with 2D integral
   N=3000;
   xrange=m(1)+6*[-1,1]*S(1,1)^0.5;
   yrange=m(2)+6*[-1 1]*S(2,2)^0.5;
   dx=diff(xrange)/(N-1);
   dy=diff(yrange)/(N-1);
   [XG,YG]=ndgrid(xrange(1)+(0:N-1)'*diff(xrange)/(N-1),yrange(1)+(0:N-1)'*diff(yrange)/(N-1));
   z=0;
   for k=1:numel(XG),
      if all([XG(k),YG(k)]>0) || all([XG(k),YG(k)]<0),
         z=z+mvnpdf([XG(k),YG(k)],m',S)*dx*dy*YG(k)^2;
      end;
   end;
   
else % check by random generator
   Nsim=100000;
   NS=300;
   msqy=0;
   
   for k=1:NS,
      x=mvnrnd(m',S,Nsim);
      msqy=msqy+sum((all(x>0,2) | all(x<0,2)).*x(:,2).^2)/Nsim;
   end;
   z=msqy/NS;
end;
toc
fprintf('z2(funtion)=%10.4f; z2(test)=%10.4f\n',z2,z);
end

      

