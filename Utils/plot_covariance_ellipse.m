function [h,slope,m]=plot_covariance_ellipse(m,cv,alpha)
% h=plot_covariance_ellipse(m,cv,alpha)
% h=plot_covariance_ellipse(X,alpha)

if length(m(:))==2,
   m=m(:);
   if nargin<3,
      alpha=[];
   end;
else
   if nargin<2,
      alpha=[];
   else
      alpha=cv;
   end;
   X=m;
   isvalid=all(~isnan(X),2);
   X=X(isvalid,:);
   m=mean(X,1)';
   cv=cov(X);
end;
if isempty(alpha),
   alpha=0.05;
end;



[V,L]=eig(cv);
L(L(:)<0)=0;
[Lmax,imax]=max(diag(L));
%  INT[exp(-0.5*x'*V*L^-1*V'*x)/2/pi/det(L)^0.5] dx    ; y=L^-0.5*V'*x ;  x=V*L^0.5*y; det(L)=prod(diag(L))
%= INT[r*exp(-0.5*r^2)] dr
%= -exp(-0.5*r^2)
%= 1-exp(-0.5*R^2)=1-alpha
%=>R^2=-2*log(alpha)
%=>R=(-2*log(alpha))^0.5;
%

R=(-2*log(alpha))^0.5;

phi=(0:360)*pi/180;
y=R*[cos(phi);sin(phi)];

x=repmat(m,1,length(phi))+V*diag(diag(L).^0.5)*y;
h=plot(x(1,:),x(2,:),'-k');


if V(1,imax)==0,
   if V(2,imax)>0,
      slope=inf;
   else
      slope=-inf;
   end;
else
   slope=V(2,imax)/V(1,imax);
end;
end
