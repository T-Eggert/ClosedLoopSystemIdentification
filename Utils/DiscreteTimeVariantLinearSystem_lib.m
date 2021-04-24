% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
% Library for computing the covariance matrix of the output of a closed-loop time-discrete linear system and its likelihood
function dls_lib=DiscreteTimeVariantLinearSystem_lib()

%  [x_sim,y_sim,r,v]=sim_noisy_system(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,Nsim,opts)
dls_lib.sim_noisy_system=@sim_noisy_system;  
%  [x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.MeanCovMatrix_DiscretLinearSystem=@MeanCovMatrix_DiscretLinearSystem;
%  [x_hat,Vx,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.expected_DiscreteSystem_stats=@expected_DiscreteSystem_stats;
%  [L,Vxx_Asymp]=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vgain,u_inf,Vrx,Vry)
dls_lib.analyse_asymptotic_state_cov=@analyse_asymptotic_state_cov;
%  [VyAsymp,Vxx_Asymp]=compute_asymptotic_Autocov_y(u_inf,A,B,C,vg,Vrx,cva_rx,Vry,maxdelay,cva_ry,isErrorClamp)
dls_lib.compute_asymptotic_Autocov_y=@compute_asymptotic_Autocov_y;
%  Vrx0=compute_asymptotic_start_Vrx(pars,Data,ci,A,B,C,Vrx,cva_rx)
dls_lib.compute_asymptotic_start_Vrx=@compute_asymptotic_start_Vrx;
%  [x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,Gv,Gr,u,y,x0,Vx0,Vrx,Vry,opts)
dls_lib.kalman_observer=@kalman_observer;
%  [NLL,w_ssqerr,logDet]=incomplete_negLogLike_fun_old(A,B,C,Gv,Gr,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.incomplete_negLogLike_fun_old=@incomplete_negLogLike_fun_old;
%  [incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,Gv,Gr,y,u,x0,Vx0,Vx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.incomplete_negLogLike_fun_new=@incomplete_negLogLike_fun_new;


% AKF=expected_AKF_moving_window_from_cv(y_hat,cv,maxdelay,HalfWindowWidth,indexRange,method)
dls_lib.expected_AKF_moving_window_from_cv=@expected_AKF_moving_window_from_cv;

%  [A,B,C,Gv,Gr,inp,x0,Vrx,cva_rx,Vry,cva_ry,VryEC]=mk_TimeVariantSystem(Data,pars)
dls_lib.mk_TimeVariantSystem=@mk_TimeVariantSystem;

dls_lib.test_loglikelihood_time_variant_system=@test_loglikelihood_time_variant_system;% [NLL,x_sim,y_sim]=test_loglikelihood_time_variant_system(x_sim,y_sim)
end


function [x_sim,y_sim,r,v]=sim_noisy_system(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,Nsim,opts)
%[x_sim,y_sim,r,v]=sim_noisy_system(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,Nsim,opts)
%
%     x(n+1)=      A(n)*x(n)+B(n)*(inp(n)+Gv(n)*v)+Gr(n)*r
%
%  For all trial types:
%                       y(n)=C(n)*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)        : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C(n-1)*x_sim(n-1)+Gv(n-1)*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + Gv(n-1)*v(n-1)  for error clamp trials at n-1.
%                              VyType=Vy                            for closed-loop
%                                    =VryEC if ~isnan(VryEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,Vx0)   : initial value of the states
%
% Arguments:
% A:           System matrix                           [dim_x,dim_x,N]
% B:           Input gain matrix                       [dim_x,dim_u,N]
% C:           Output gain matrix                      [dim_y,dim_x,N]
% Gv:          transfer gain(s) for measurement noise  [dim_y,N]
% Gr:          gain matrix of planning noise           [dim_x,dim_x,N]
% inp:         Input for t=(0:N-1)'*dt [N,dim_u]
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation of the state noise for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       Covariance matrix of the measurement noise for error clamp trials [dim_y,dim_y]
% Nsim:        Number of simulations
%
% opts:        optional structure with the following fields:
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                              (default: zeros(N,1)  )
%                noise_method: ==0: covariance matrix of signal dependent noise is  diag(cva_rx.^2.*x_sim(n).^2)
%                              ~=0:                     "                           diag(cva_rx.^2.*(diag(Vx_expected(n))+x_expected(n).^2));
%                                   (default=1)
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
% x_sim       : The simulated state sequence for t=(0:N)'*dt           [N+1,Nsim*dim_x]   
% y_sim       : The simulated output for t=(0:N)'*dt                     [N,Nsim*dim_y]
%
%
% copyright: T. Eggert  (2020)
%            Ludwig-Maximilians Universität, München
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried,Germany
%            e-mail: eggert@lrz.uni-muenchen.de
x0=x0(:);
dim_y=size(C,1);
dim_u=size(B,2);

if dim_y~=dim_u
   error('size(B,2) ~= size(C,1)!');
end

if size(inp,1)==1 && dim_u==1
   inp=inp';
end

N=size(inp,1);
dim_x=size(A,1);
if length(size(A))==2
   A=reshape(repmat(A,1,N),[size(A),N]);
end
if length(size(B))==2
   B=reshape(repmat(B,1,N),[size(B),N]);
end
if length(size(C))==2
   C=reshape(repmat(C,1,N),[size(C),N]);
end
if size(Gv,1)==N && size(Gv,2)==dim_y
   Gv=Gv';
end
if size(Gv,2)==1
   Gv=repmat(Gv,1,N);
end
if length(size(Gr))==2
   Gr=reshape(repmat(Gr,1,N),[size(Gr),N]);
end

if nargin<13 || isempty(VryEC)
   VryEC=NaN;
end
if isnan(VryEC(1,1))
   VryEC=Vry;
end

if nargin<14 || isempty(Nsim)
   Nsim=100;
end
cva_rx=abs(cva_rx(:));

default_opts.TrialType=zeros(N,1);
default_opts.noise_method=1;
default_opts.signed_EDPMN=false;



if nargin<15 || isempty(opts)
   opts=[];
end
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>1)
   error('TrialType may not exceed 1!');
end
if any(opts.TrialType<0)
   error('TrialType must exceed -1!');
end


if length(opts.TrialType)~=N
   error('length(opts.TrialType) must equal the number of observations!');
end

x_sim=zeros(N+1,Nsim*dim_x);  % t=(0:N)'*dt;
y_sim=zeros(N,Nsim*dim_y);  


%** compute the expected state, the expected state covariance matrix, and the covariance matrix of the state noise

if all(cva_rx==0) && all(cva_ry==0)
   Vrx_n=reshape(repmat(Vrx,1,N),dim_x,dim_x,N);
   Vry_n=reshape(repmat(Vry,1,N),dim_y,dim_y,N);
   is_EC=(opts.TrialType==1);
   if any(is_EC)
      N_EC=sum(is_EC);
      Vry_n(:,:,is_EC)=repmat(VryEC,1,1,N_EC);
   end
elseif opts.noise_method==0
   Vrx_n=zeros(dim_x,dim_x,N);
   Vry_n=zeros(dim_y,dim_y,N);
else
   Vrx_n=zeros(dim_x,dim_x,N);
   Vrx_n(:,:,1)=Vrx+diag(cva_rx.^2.*(diag(Vx0)+x0.^2));
   
   Vry_n=zeros(dim_y,dim_y,N);
   if opts.TrialType(1)==1
      Vry_n(:,:,1)=VryEC;
   else
      Vry_n(:,:,1)=Vry;
   end
   
   
   x_expected=x0;
   Vx_expected=Vx0;
   for k=2:N
      
      if opts.TrialType(k)==0  % closed loop trial
         VyType=Vry;
      else
         VyType=VryEC;
      end
      if any(cva_ry>0)
         % update the motor noise covariance matrix
         if opts.TrialType(k-1)==0  % closed loop trial
            err_nm1_expected=inp(k-1,:)'-C(:,:,k-1)*x_expected;
            if opts.signed_EDPMN
               expected_err_sq=zeros(size(inp,2));
               cv_expected_cx=C(:,:,k-1)*Vx_expected*C(:,:,k-1)';
               cv_expected_err=cv_expected_cx+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
               for inp_i=1:size(inp,2)
                  m_cx_err=[C(inp_i,:,k-1)*x_expected;err_nm1_expected(inp_i)];
                  S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
                  expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
               end
            else
               expected_err_sq=C(:,:,k-1)*Vx_expected*C(:,:,k-1)'+ err_nm1_expected*err_nm1_expected'+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            end
         else
            if opts.signed_EDPMN
               expected_err_sq=zeros(size(inp,2));
               cv_expected_cx=C(:,:,k-1)*Vx_expected*C(:,:,k-1)';
               cv_expected_err=diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
               for inp_i=1:size(inp,2)
                  m_cx_err=[C(inp_i,:,k-1)*x_expected;inp(k-1,inp_i)];
                  S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
                  expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
               end
            else
               expected_err_sq=inp(k-1,:)'*inp(k-1,:) +diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            end
         end
         Vry_n(:,:,k)=VyType+diag(cva_ry.^2.*diag(expected_err_sq));
      else
         Vry_n(:,:,k)=VyType;
      end
      
      
      x_expected=A(:,:,k-1)*x_expected+B(:,:,k-1)*inp(k-1,:)';
      Vx_expected=A(:,:,k-1)*Vx_expected*A(:,:,k-1)'+Gr(:,:,k-1)*Vrx_n(:,:,k-1)*Gr(:,:,k-1)' + B(:,:,k-1)*diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1))*B(:,:,k-1)';
      
        %** update the state covariance matrix
      if any(cva_rx>0)
         Vrx_n(:,:,k)=Vrx+diag(cva_rx.^2.*(diag(Vx_expected)+x_expected.^2));
      else
         Vrx_n(:,:,k)=Vrx;
      end
   end
      
   
end


for si=1:Nsim
   if all(cva_ry==0)
      v=mvnrnd(zeros(1,dim_y),Vry,N);
      is_EC=(opts.TrialType==1);
      if any(is_EC)
         N_EC=sum(is_EC);
         v(is_EC,:)=mvnrnd(zeros(1,dim_y),VryEC,N_EC);
      end
   else
      v=zeros(N,dim_y);
      if opts.noise_method==0
         if opts.TrialType(1)==1
            Vry_n(:,:,1)=VryEC;
         else
            Vry_n(:,:,1)=Vry;
         end
      end
      v(1,:)=mvnrnd(zeros(1,dim_y),Vry_n(:,:,1),1);
   end
   
   xbar=x0+mvnrnd(zeros(1,dim_x),Vx0,1)';
   if all(cva_rx==0)
      r=mvnrnd(zeros(1,dim_x),Vrx,N);
   else
      r=zeros(N,dim_x);
      if opts.noise_method==0
         Vrx_n(:,:,1)=Vrx+diag(cva_rx.^2.*xbar.^2);
      end
      r(1,:)=mvnrnd(zeros(1,dim_x),Vrx_n(:,:,1),1);
   end
   
   
   x=zeros(N+1,dim_x);
   y=zeros(N,dim_y);
   x(1,:)=xbar';
   y(1,:)=(C(:,:,1)*xbar+v(1,:)')';
   for k=2:N
      if opts.TrialType(k)==0  % closed loop trial
         VyType=Vry;
      else
         VyType=VryEC;
      end
      if opts.TrialType(k-1)==0      % closed loop trial
         err_nm1=inp(k-1,:)'+diag(Gv(:,k-1))*v(k-1,:)'-C(:,:,k-1)*xbar;
      elseif opts.TrialType(k-1)==1  % error clamp trial
         err_nm1=inp(k-1,:)'+diag(Gv(:,k-1))*v(k-1,:)';
      end
      xbar=A(:,:,k-1)*xbar+B(:,:,k-1)*(inp(k-1,:)'+diag(Gv(:,k-1))*v(k-1,:)')+Gr(:,:,k-1)*r(k-1,:)';
      
      if any(cva_ry>0)
         if opts.noise_method==0
            if ~opts.signed_EDPMN || err_nm1*(C(:,:,k-1)*x(k-1,:)')>0
               Vry_n(:,:,k)=VyType+diag(cva_ry.^2.*err_nm1.^2);
            else
               Vry_n(:,:,k)=VyType;
            end
         end
         v(k,:)=mvnrnd(zeros(1,dim_y),Vry_n(:,:,k),1);
      end
      
      if any(cva_rx>0)
         if opts.noise_method==0
            Vrx_n(:,:,k)=Vrx+diag(cva_rx.^2.*xbar.^2);
         end
         r(k,:)=mvnrnd(zeros(1,dim_x),Vrx_n(:,:,k),1);
      end
      x(k,:)=xbar';
      y(k,:)=(C(:,:,k)*xbar+v(k,:)')';
   end
   x(N+1,:)=(A(:,:,N)*xbar+B(:,:,N)*(inp(N,:)'+diag(Gv(:,N))*v(N,:)')+Gr(:,:,N)*r(N,:)')';
   
   
   x_sim(:,(si-1)*dim_x+(1:dim_x))=x; 
   y_sim(:,(si-1)*dim_y+(1:dim_y))=y; 
end
end

function [x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%[x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%
%
%     x(n+1)=      A(n)*x(n)+B(n)*(inp(n)+Gv(n)*v)+Gr(n)*r
%
%  For all trial types:
%                       y(n)=C(n)*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)        : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C(n-1)*x_sim(n-1)+Gv(n-1)*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + Gv(n-1)*v(n-1)  for error clamp trials at n-1.
%                              VyType=Vy                            for closed-loop
%                                    =VryEC if ~isnan(VryEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,Vx0)   : initial value of the states
%
% Arguments:
% A:           System matrix                           [dim_x,dim_x,N]
% B:           Input gain matrix                       [dim_x,dim_u,N]
% C:           Output gain matrix                      [dim_y,dim_x,N]
% Gv:          transfer gain(s) for measurement noise  [dim_y,N]
% Gr:          gain matrix of planning noise           [dim_x,dim_x,N]
% inp:         Input for t=(0:N-1)'*dt [N,dim_u]
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation of the state noise for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       Covariance matrix of the measurement noise for error clamp trials [dim_y,dim_y]
%
% opts:        optional structure with the following fields:
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                              (default: zeros(N,1)  )
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
%
% Returned values:
% x_hat       : The expected state sequence         [N+1,dim_x]      ( t=(0:N)'*dt )
% y_hat       : The expected output                   [N,dim_y]
% VyM         : The covariance matrix of the output   [N*dim_y,N*dim_y]
%
%
% copyright: T. Eggert  (2020)
%            Ludwig-Maximilians Universität, München
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried,Germany
%            e-mail: eggert@lrz.uni-muenchen.de

x0=x0(:);
dim_y=size(C,1);
dim_u=size(B,2);

if dim_y~=dim_u
   error('size(B,2) ~= size(C,1)!');
end

if size(inp,1)==1 && dim_u==1
   inp=inp';
end

N=size(inp,1);
dim_x=size(A,1);

if length(size(A))==2
   A=reshape(repmat(A,1,N),[size(A),N]);
end
if length(size(B))==2
   B=reshape(repmat(B,1,N),[size(B),N]);
end
if length(size(C))==2
   C=reshape(repmat(C,1,N),[size(C),N]);
end
if size(Gv,1)==N && size(Gv,2)==dim_y
   Gv=Gv';
end
if size(Gv,2)==1
   Gv=repmat(Gv,1,N);
end
if length(size(Gr))==2
   Gr=reshape(repmat(Gr,1,N),[size(Gr),N]);
end


if any(any(isnan(inp)))
   error('no missings are allowed in inp!');
end

cva_rx=cva_rx(:);
cva_ry=cva_ry(:);
default_opts.TrialType=zeros(N,1);
default_opts.signed_EDPMN=false;



if nargin<13 || isempty(VryEC)
   VryEC=NaN;
end
if isnan(VryEC(1,1))
   VryEC=Vry;
end

if nargin<14 || isempty(opts)
   opts=[];
end
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>1)
   error('TrialType may not exceed 1!');
end
if any(opts.TrialType<0)
   error('TrialType must exceed -1!');
end


if length(opts.TrialType)~=N
   error('length(opts.TrialType) must equal the number of inputs!');
end



Dn=Gr;
An=A;
Bn=B;
Zn=inp';

% compute the expected output and its covariance matrix
y_hat=zeros(N,dim_y);        % the expected output
Vx=zeros(dim_x,dim_x,N+1);   % the covariance matrix of the states
Vy=zeros(dim_y,dim_y,N);     % the covariance matrix of the output
Vry_n=zeros(dim_y,dim_y,N);  % the covariance matrix of the motor noise

x_hat=zeros(N+1,dim_x);
x_hat(1,:)=x0';
y_hat(1,:)=x0'*C(:,:,1)';
xbar=x0;

if opts.TrialType(1)==0
   Vry_n(:,:,1)=Vry;
else
   Vry_n(:,:,1)=VryEC;
end
Vx(:,:,1)=Vx0;
Vy(:,:,1)=C(:,:,1)*Vx(:,:,1)*C(:,:,1)'+Vry_n(:,:,1);
for k=2:N
   if opts.TrialType(k)==0  % closed loop trial
      VryType=Vry;
   else
      VryType=VryEC;
   end
   
   if any(cva_ry>0)
      
      % update the motor noise covariance matrix
      if opts.TrialType(k-1)==0  % closed loop trial
         err_nm1_expected=inp(k-1,:)'-C(:,:,k-1)*xbar;
         if opts.signed_EDPMN
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)';
            cv_expected_err=cv_expected_cx+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            for inp_i=1:size(inp,2)
               m_cx_err=[C(inp_i,:,k-1)*xbar;err_nm1_expected(inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end
         else
            expected_err_sq=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)'+ err_nm1_expected*err_nm1_expected'+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
         end
      else
         if opts.signed_EDPMN
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)';
            cv_expected_err=diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            for inp_i=1:size(inp,2)
               m_cx_err=[C(inp_i,:,k-1)*xbar;inp(k-1,inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end
         else
            expected_err_sq=inp(k-1,:)'*inp(k-1,:) +diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
         end
      end
      Vry_n(:,:,k)=VryType+diag(cva_ry.^2.*diag(expected_err_sq));
   else
      Vry_n(:,:,k)=VryType;
   end
   
   
   Vrx_n=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,k-1))+xbar.^2));
   xbar=An(:,:,k-1)*xbar+Bn(:,:,k-1)*Zn(:,k-1);
   Vx_tmp=An(:,:,k-1)*Vx(:,:,k-1)*An(:,:,k-1)'+Dn(:,:,k-1)*Vrx_n*Dn(:,:,k-1)' + Bn(:,:,k-1)*diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1))*Bn(:,:,k-1)';
   %** clip Vx: ****
   Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
   Vx(:,:,k)=Vx_tmp;
   
   
   x_hat(k,:)=xbar';
   y_hat(k,:)=xbar'*C(:,:,k)';
   
   Vy(:,:,k)=C(:,:,k)*Vx(:,:,k)*C(:,:,k)'+Vry_n(:,:,k);
end

Vrx_n=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,N))+xbar.^2));
xbar=An(:,:,N)*xbar+Bn(:,:,N)*Zn(:,N);
Vx_tmp=An(:,:,N)*Vx(:,:,N)*An(:,:,N)'+Dn(:,:,N)*Vrx_n*Dn(:,:,N)' + Bn(:,:,N)*diag(Gv(:,N))*Vry_n(:,:,N)*diag(Gv(:,N))*Bn(:,:,N)';
%** clip Vx: ****
Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
Vx(:,:,N+1)=Vx_tmp;


x_hat(N+1,:)=xbar';



% compute the full covariance matrix of y:
VyM=zeros(N*dim_y, N*dim_y);
for j=1:N-1
   Vij=Vx(:,:,j);
   VyM((j-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))=C(:,:,j)*Vij*C(:,:,j)'+Vry_n(:,:,j);
   
   v_ij=Bn(:,:,j)*diag(Gv(:,j))*Vry_n(:,:,j);
      
   for i=j+1:N
      
      Vij=An(:,:,i-1)*Vij;
      
%       COV_X(:,:,i,j)=Vij;  % the cross covariance matrices may be stored for later use
%       COV_X(:,:,j,i)=Vij';
      if i>j+1
         v_ij=An(:,:,i-1)*v_ij;
      end
      VyM((i-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))=C(:,:,i)*(Vij*C(:,:,j)'+v_ij);
      VyM((j-1)*dim_y+(1:dim_y),(i-1)*dim_y+(1:dim_y))=VyM((i-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))';
   end
end
VyM((N-1)*dim_y+(1:dim_y),(N-1)*dim_y+(1:dim_y))=C(:,:,N)*Vx(:,:,N)*C(:,:,N)'+Vry_n(:,:,N);

end


function [x_hat,Vx,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
% this is a copy of the first loop of MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%     x(n+1)=      A(n)*x(n)+B(n)*(inp(n)+Gv(n)*v)+Gr(n)*r
%
%  For all trial types:
%                       y(n)=C(n)*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)        : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C(n-1)*x_sim(n-1)+Gv(n-1)*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + Gv(n-1)*v(n-1)  for error clamp trials at n-1.
%                              VyType=Vy                            for closed-loop
%                                    =VryEC if ~isnan(VryEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,Vx0)   : initial value of the states
%
% Arguments:
% A:           System matrix                           [dim_x,dim_x,N]
% B:           Input gain matrix                       [dim_x,dim_u,N]
% C:           Output gain matrix                      [dim_y,dim_x,N]
% Gv:          transfer gain(s) for measurement noise  [dim_y,N]
% Gr:          gain matrix of planning noise           [dim_x,dim_x,N]
% inp:         Input for t=(0:N-1)'*dt [N,dim_u]
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation of the state noise for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       Covariance matrix of the measurement noise for error clamp trials [dim_y,dim_y]
%
% opts:        optional structure with the following fields:
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                              (default: zeros(N,1)  )
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
%
% Returned values:
% x_hat       : The expected state sequence              [N+1,dim_x]      ( t=(0:N)'*dt )
% Vx          : covariance matrix of the states          [dim_x,dim_x,N+1]
% Vrx_n       : covariance matrix of the planning noise  [dim_x,dim_x,N]
% Vry_n       : covarince matrix of the motor noise      [dim_y,dim_y,N]
%
%
% copyright: T. Eggert  (2020)
%            Ludwig-Maximilians Universität, München
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried,Germany
%            e-mail: eggert@lrz.uni-muenchen.de

x0=x0(:);
dim_y=size(C,1);
dim_u=size(B,2);

if dim_y~=dim_u
   error('size(B,2) ~= size(C,1)!');
end

if size(inp,1)==1 && dim_u==1
   inp=inp';
end

N=size(inp,1);
dim_x=size(A,1);

if length(size(A))==2
   A=reshape(repmat(A,1,N),[size(A),N]);
end
if length(size(B))==2
   B=reshape(repmat(B,1,N),[size(B),N]);
end
if length(size(C))==2
   C=reshape(repmat(C,1,N),[size(C),N]);
end
if size(Gv,1)==N && size(Gv,2)==dim_y
   Gv=Gv';
end
if size(Gv,2)==1
   Gv=repmat(Gv,1,N);
end
if length(size(Gr))==2
   Gr=reshape(repmat(Gr,1,N),[size(Gr),N]);
end

if any(any(isnan(inp)))
   error('no missings are allowed in inp!');
end

cva_rx=cva_rx(:);
cva_ry=cva_ry(:);
default_opts.TrialType=zeros(N,1);
default_opts.signed_EDPMN=false;



if nargin<13 || isempty(VryEC)
   VryEC=NaN;
end
if isnan(VryEC(1,1))
   VryEC=Vry;
end

if nargin<14 || isempty(opts)
   opts=[];
end
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>1)
   error('TrialType may not exceed 1!');
end
if any(opts.TrialType<0)
   error('TrialType must exceed -1!');
end


if length(opts.TrialType)~=N
   error('length(opts.TrialType) must equal the number of inputs!');
end




Dn=Gr;
An=A;
Bn=B;
Zn=inp';

% compute the expected output and its covariance matrix
y_hat=zeros(N,dim_y);      % the expected output
Vx=zeros(dim_x,dim_x,N+1);   % the covariance matrix of the state noise
Vy=zeros(dim_y,dim_y,N);   % the covariance matrix of the output
Vry_n=zeros(dim_y,dim_y,N);% the covariance matrix of the motor noise
Vrx_n=zeros(dim_x,dim_x,N);% the covariance matrix of the state noise

x_hat=zeros(N+1,dim_x);
x_hat(1,:)=x0';
y_hat(1,:)=x0'*C(:,:,1)';
xbar=x0;

if opts.TrialType(1)==0
   Vry_n(:,:,1)=Vry;
else
   Vry_n(:,:,1)=VryEC;
end
Vx(:,:,1)=Vx0;
Vy(:,:,1)=C(:,:,1)*Vx(:,:,1)*C(:,:,1)'+Vry_n(:,:,1);
for k=2:N
   if opts.TrialType(k)==0  % closed loop trial
      VryType=Vry;
   else
      VryType=VryEC;
   end
   
   if any(cva_ry>0)
      
      % update the motor noise covariance matrix
      if opts.TrialType(k-1)==0  % closed loop trial
         err_nm1_expected=inp(k-1,:)'-C(:,:,k-1)*xbar;
         if opts.signed_EDPMN
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)';
            cv_expected_err=cv_expected_cx+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            for inp_i=1:size(inp,2)
               m_cx_err=[C(inp_i,:,k-1)*xbar;err_nm1_expected(inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end
         else
            expected_err_sq=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)'+ err_nm1_expected*err_nm1_expected'+diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
         end
      else
         if opts.signed_EDPMN
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C(:,:,k-1)*Vx(:,:,k-1)*C(:,:,k-1)';
            cv_expected_err=diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
            for inp_i=1:size(inp,2)
               m_cx_err=[C(inp_i,:,k-1)*xbar;inp(k-1,inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end
         else
            expected_err_sq=inp(k-1,:)'*inp(k-1,:) +diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1));
         end
      end
      Vry_n(:,:,k)=VryType+diag(cva_ry.^2.*diag(expected_err_sq));
   else
      Vry_n(:,:,k)=VryType;
   end
   
   
   Vrx_n(:,:,k-1)=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,k-1))+xbar.^2));
   xbar=An(:,:,k-1)*xbar+Bn(:,:,k-1)*Zn(:,k-1);
   Vx_tmp=An(:,:,k-1)*Vx(:,:,k-1)*An(:,:,k-1)'+Dn(:,:,k-1)*Vrx_n(:,:,k-1)*Dn(:,:,k-1)' + Bn(:,:,k-1)*diag(Gv(:,k-1))*Vry_n(:,:,k-1)*diag(Gv(:,k-1))*Bn(:,:,k-1)';
   %** clip Vx: ****
   Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
   Vx(:,:,k)=Vx_tmp;
   
   x_hat(k,:)=xbar';
   y_hat(k,:)=xbar'*C(:,:,k)';
   
   Vy(:,:,k)=C(:,:,k)*Vx(:,:,k)*C(:,:,k)'+Vry_n(:,:,k);
end

Vrx_n(:,:,N)=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,N))+xbar.^2));
xbar=An(:,:,N)*xbar+Bn(:,:,N)*Zn(:,N);
Vx_tmp=An(:,:,N)*Vx(:,:,N)*An(:,:,N)'+Dn(:,:,N)*Vrx_n(:,:,N)*Dn(:,:,N)' + Bn(:,:,N)*diag(Gv(:,N))*Vry_n(:,:,N)*diag(Gv(:,N))*Bn(:,:,N)';
%** clip Vx: ****
Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
Vx(:,:,N+1)=Vx_tmp;


x_hat(N+1,:)=xbar';
end


function [L,Vxx_Asymp]=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vgain,u_inf,Vrx,Vry)
%** analyse the asymptotic stability of the noise for the time discrete dynamics: **
%**   Vrx_n=Vrx+diag(cva_rx.^2.*(diag(Vx)+x_inf.^2));
%**   Vx(n+1)=A*Vx(n)*A'+Vrx_n + B*diag(vgain.^2)*diag(S2_ry(:,n))*B';
%**      with 
%**   S2_ry(:,n+1)= S2_ry_0+cva_ry.^2*(  (u_inf-C*x_inf).^2 + diag(C*Vx(n)*C') + vgain.^2.*S2_ry(:,n)  ) for closed-loop trials  (isErrorClamp==false)
%**               = S2_ry_0+cva_ry.^2*(   u_inf.^2 + vgain.^2.*S2_ry(:,n)  )                             for error-clamp trials  (isErrorClamp==true)
%      S2_ry_0=diag(Vry)
%      C: ones(dim_y,dim_x)
%     
%
% Arguments:
%            A: system matrices defining the retention
%       cva_rx: vector containing the coefficients of variation of the state noise
%               If these two parameters are the only ones to be passed, the function assumes that error-dependent motor noise is absent
%               and computes only L
%
%       cva_ry: vector containing the coefficients of variation of the motor noise
%               If this vector is empty or if all(cva_ry==0), error-dependent motor noise is absent.
%               Otherwise, isErrorClamp,B,and vgain must be provided
%               default: []
% isErrorClamp: true: error-dependent motor noise is computed for error-clamp trials. u_inf is the stationary error-clamp value
%               Irrelevant for isempty(cva_ry) || all(cva_ry==0)
%            B: system matrices defining the input sensitivity
%        vgain: transfer gain of measurement noise to the error signal 
%
%        All following arguments are irrelevant and do not have to be defined for nargout<2:
%  u_inf: constant input 
%    Vrx: state covariance matrix of constant state noise
%    Vry: covariance matrix of constant measurement noise (assumed to be diagonal) 

if nargin<3 || (~isempty(cva_ry) && all(cva_ry==0))
   cva_ry=[];
end

if isempty(cva_ry)
   M=get_noise_diffEQ_sysmatrix(A,cva_rx);
else
   dim_y=length(cva_ry);
   M=get_noise_diffEQ_sysmatrix_with_ErrorDependentMotorNoise(A,cva_rx,B,vgain,cva_ry,isErrorClamp);
end



[V,L]=eig(M);
L=diag(L);
if nargout<2
   return;
end

dim_x=size(A,1);
if any(abs(L)>=1)
   % instable: reduce cva_rx until stability is achieved
   while any(cva_rx>=1e-5) || (~isempty(cva_ry) && any(cva_ry>=1e-5) )
      for k=1:dim_x
         if cva_rx(k)>=1e-5
            cva_rx(k)=cva_rx(k)*0.8;
         else
            cva_rx(k)=0;
         end
      end
      
      if ~isempty(cva_ry) && any(cva_ry>=1e-5)
         for k=1:dim_y
            if cva_ry(k)>=1e-5
               cva_ry(k)=cva_ry(k)*0.8;
            else
               cva_ry(k)=0;
            end
         end
      end
      if isempty(cva_ry)
         L=analyse_asymptotic_state_cov(A,cva_rx);
      else
         L=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vgain);
      end
      if all(abs(L)<1)
         break;
      end
   end
   if any(abs(L)>=1)
      Vxx_Asymp=inf(dim_x,dim_x);
      return;
   end
   if isempty(cva_ry)
      M=get_noise_diffEQ_sysmatrix(A,cva_rx);
   else
      M=get_noise_diffEQ_sysmatrix_with_ErrorDependentMotorNoise(A,cva_rx,B,vgain,cva_ry,isErrorClamp);
   end
end

vgain=vgain(:);
dim_u=length(vgain);
if dim_u~=size(B,2)
   error('invalid dimension of vgain!!');
end

dim_v=round(dim_x*(dim_x+1)/2);

if isempty(cva_ry)
   VTrans=B*diag(vgain)*Vry*diag(vgain)*B';
else
   S2_ry_0=diag(Vry);
   VTrans=zeros(dim_x,dim_x);
end

x_inf=(eye(dim_x)-A)\B*u_inf;

if isempty(cva_ry)
   MC=zeros(dim_v,1);
else
   MC=zeros(dim_v+dim_y,1);
end

v1=0;
for i1=1:dim_x
   for j1=i1:dim_x
      v1=v1+1;
      if i1==j1
         MC(v1)=MC(v1)+cva_rx(i1)^2*x_inf(i1)^2;
      end
      MC(v1)=MC(v1)+Vrx(i1,j1)+VTrans(i1,j1);
   end
end

if ~isempty(cva_ry)
   for i1=1:dim_y
      if isErrorClamp
         MC(dim_v+i1)=S2_ry_0(i1)+cva_ry(i1)^2*u_inf(i1)^2;
      else
         MC(dim_v+i1)=S2_ry_0(i1)+cva_ry(i1)^2*(u_inf(i1)-sum(x_inf))^2;
      end
   end
end


vs=(eye(size(M,1))-M)\MC;
%vs-M*vs-MC
v1=0;
Vxx_Asymp=zeros(dim_x,dim_x);
for i1=1:dim_x
   for j1=i1:dim_x
      v1=v1+1;
      Vxx_Asymp(i1,j1)=vs(v1);
      if i1~=j1
         Vxx_Asymp(j1,i1)=vs(v1);
      end
   end
end
end



function M=get_noise_diffEQ_sysmatrix(A,cva_rx)
% M returns the system matrix for the state noise covariance matrix in the normal form.
%    This is a internal (non-exported) utility only called by analyse_asymptotic_state_cov 
dim_x=size(A,1);
dim_v=round(dim_x*(dim_x+1)/2);
M=zeros(dim_v,dim_v);

v1=0;
for i1=1:dim_x,
   for j1=i1:dim_x,
      v1=v1+1;
      
      v2=0;
      for i2=1:dim_x,
         for j2=i2:dim_x,
            v2=v2+1;
            M(v1,v2)=M(v1,v2)+A(i1,i2)*A(j1,j2);
            if i2~=j2,
               M(v1,v2)=M(v1,v2)+A(i1,j2)*A(j1,i2);
            end;
            if i1==j1 && i2==i1 && j2==j1,
               M(v1,v2)=M(v1,v2)+cva_rx(i1)^2;
            end;
         end;
      end;
      
   end;
end;
end



function M=get_noise_diffEQ_sysmatrix_with_ErrorDependentMotorNoise(A,cva_rx,B,vgain,cva_ry,isErrorClamp)
% M returns the system matrix for the state noise covariance matrix and the variances of the motor noise in the normal form.
%    This is a internal (non-exported) utility only called by analyse_asymptotic_state_cov 

dim_x=size(A,1);
dim_v=round(dim_x*(dim_x+1)/2);
dim_y=length(cva_ry);
if dim_y~=size(B,2),
   error('B has invalid width!');
end;
M=zeros(dim_v+dim_y,dim_v+dim_y);

v1=0;
for i1=1:dim_x,
   for j1=i1:dim_x,
      v1=v1+1;
      
      v2=0;
      for i2=1:dim_x,
         for j2=i2:dim_x,
            v2=v2+1;
            M(v1,v2)=M(v1,v2)+A(i1,i2)*A(j1,j2);
            if i2~=j2,
               M(v1,v2)=M(v1,v2)+A(i1,j2)*A(j1,i2);
            end;
            if i1==j1 && i2==i1 && j2==j1,
               M(v1,v2)=M(v1,v2)+cva_rx(i1)^2;
            end;
         end;
      end;
      
   end;
end;

v1=0;
for i1=1:dim_x,
   for j1=i1:dim_x,
      v1=v1+1;
      
      for i2=1:dim_y,
         M(v1,dim_v+i2)=B(i1,i2)*vgain(i2)^2*B(j1,i2);
      end;
   end;
end;

if ~isErrorClamp,
   for i1=1:dim_y,
      cva_ry_i=cva_ry(i1)^2;
      
      v2=0;
      for i2=1:dim_x,
         for j2=i2:dim_x,
            v2=v2+1;
            if i2~=j2,
               M(dim_v+i1,v2)=2*cva_ry_i;  % 2*C(i1,i2)*C(i1,j2)   we assume here that all(C(:)==1)
            else
               M(dim_v+i1,v2)=cva_ry_i;    % C(i1,i2)*C(i1,i2)
            end;
            
         end;
      end;
      
   end;
end;

for i1=1:dim_y,
   M(dim_v+i1,dim_v+i1)=cva_ry(i1)^2*vgain(i1)^2;
end;

end


function [VyAsymp,Vxx_Asymp]=compute_asymptotic_Autocov_y(u_inf,A,B,C,vg,Vrx,cva_rx,Vry,maxdelay,cva_ry,isErrorClamp)
if nargin<10
   cva_ry=[];
   isErrorClamp=[];
end;

if any(cva_rx~=0) || (~isempty(cva_ry) && any(cva_ry~=0)),
   if ~isempty(cva_ry) && any(cva_ry~=0) && any(C(:)~=1),
      error('any(C(:)~=1) is not implemented for error-dependent motor noise!!');
   end;
   [L,Vxx_Asymp]=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vg,u_inf,Vrx,Vry);
else

   [Va,La]=eig(A);
   VaI=Va^-1;
   La=diag(La);
   Vtrans=B*vg*Vry*vg*B';
   % Vxx_Asymp=A*Vxx_Asymp*A'+VTrans+Vrx
   % Vxx_Asymp-Va*La*VaI*Vxx_Asymp*VaI'*La*Va'=VTrans+Vrx                 % <= A=Va*La*VaI
   % VaI*Vxx_Asymp*VaI'-La*VaI*Vxx_Asymp*VaI'*La=VaI*(VTrans+Vrx)*VaI'    % <=   VaI*(...)*VaI'
   %    subst: Z:=VaI*Vxx_Asymp*VaI';   K:=VaI*(VTrans+Vrx)*VaI'
   % => Z-La*Z*La = K
   % => Z.*(1-diag(La)*diag(La)') = K
   % => Z=K./(1-diag(La)*diag(La)')
   % => Vxx_Asymp=Va*(  K./(1-diag(La)*diag(La)')*Va'
   % => Vxx_Asymp=Va*( (VaI*(VTrans+Vrx)*VaI')./(1-diag(La)*diag(La)')*Va'
   Vxx_Asymp=Va*( (VaI*(Vrx+Vtrans)*VaI')./(1-La*La'))*Va';
end;

if Vxx_Asymp(1,1)<inf,
   if any(any(isinf(Vxx_Asymp))) || any(any(isnan(Vxx_Asymp))),
      disp('*');
   end;
   [Va,La]=eig(Vxx_Asymp);
   La=max(real(diag(La)),0);  % symmetric matrices have real eigenvalues
   Vxx_Asymp=Va*diag(La)*Va';
end;

VyAsymp=zeros(1,1+maxdelay);
VyAsymp(1)=C*Vxx_Asymp*C' + Vry;
vect_ytrans=B*vg*Vry;
for k=2:1+maxdelay,
   
   Vxx_Asymp=A*Vxx_Asymp;
   VyAsymp(k)=C*Vxx_Asymp*C' + C*vect_ytrans;
   vect_ytrans=A*vect_ytrans;
end;

end

function Vrx0=compute_asymptotic_start_Vrx(pars,Data,ci,A,B,C,Vrx,cva_rx)
% computes the asymptotic state variance for the beginning of the trial sequence specified by Data
% (index: ci) and the parameter set pars
% Arguments:
% pars: parameter structure with the fields as defined in fit_likelihoodDiscreteLinearSystem
%             required fields are:
%                      pars.Vry
%                      pars.vgain_EC
%                      pars.vgain_CL
%                      pars.cva_ry
%                      pars.VryEC
%
% Data: structure with the fileds
%           TrialType: [NTrials,NSequence]
%           InputType: 0 or 1 as defined in MeanCovMatrix_DiscretLinearSystem
%                   u: [NTrials,NSequence] visuomotor distortion
% ci  :  column index in TrialType and u
% A,B,C,Vrx,cva_rx: optional parameters which will be generated from pars if not provided


if nargin<5 || isempty(B),
   B=[pars.bf;pars.bs];
end;
if nargin<4 || isempty(A),
   if Data.InputType==1,
      A=diag([pars.af;pars.as]);
   else
      A=diag([pars.af;pars.as])-repmat(B,1,2);
   end;
end;
if nargin<6 || isempty(C),
   C=[1,1];
end;
if nargin<7 || isempty(Vrx),
   Vrx=[pars.Vrx11,pars.Vrx12;pars.Vrx12,pars.Vrx22];
end;
if nargin<8 || isempty(cva_rx),
   cva_rx=[pars.cva_xf;pars.cva_xs];
end;


if isnan(pars.VryEC(1,1)),
   VryEC=pars.Vry;
else
   VryEC=pars.VryEC;
end;

if Data.InputType==1,
   u_inf=0;
   A_inf=A;
   if any(Data.TrialType(1,ci)==[1 3]),
      vgain_inf=pars.vgain_EC;
      isErrorClamp_inf=true;
      Vry_inf=VryEC;
   else
      vgain_inf=pars.vgain_CL;
      isErrorClamp_inf=false;
      Vry_inf=pars.Vry;
   end;
else
   if any(Data.TrialType(1,ci)==[1 3]),
      u_inf=0;
      A_inf=A+B*C;
      vgain_inf=pars.vgain_EC;
      isErrorClamp_inf=true;
      Vry_inf=VryEC;
   else
      u_inf=Data.u(1,ci);
      A_inf=A;
      vgain_inf=pars.vgain_CL;
      isErrorClamp_inf=false;
      Vry_inf=pars.Vry;
   end
end
[tmp,Vrx0]=compute_asymptotic_Autocov_y(u_inf,A_inf,B,C,vgain_inf,Vrx,cva_rx,Vry_inf,0,pars.cva_ry,isErrorClamp_inf);
end


%                                                                
function [x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,Gv,Gr,u,y,x0,Vx0,Vrx,Vry,opts)
%  [x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,Gv,Gr,u,y,x0,Vx0,Vrx,Vry,opts)
% Kalman filter and Kalman smoother for the system with the following trial types:
%
%     x(n+1)=      A(n)*x(n)+B(n)*(u(n)+Gv(n)*v)+Gr(n)*r
%
%  For all trial types:
%                       y(n)=C(n)*x(n) + v
%                       r(n) ~ N(0,Vrx(n))        : process noise
%                       v(n) ~ N(0,Vry(n))        : measurement noise
%                       x0~ N(0,Vx0)   : initial value of the states
%
% Arguments:
% A:           System matrix                           [dim_x,dim_x,N]
% B:           Input gain matrix                       [dim_x,dim_u,N]
% C:           Output gain matrix                      [dim_y,dim_x,N]
% Gv:          transfer gain(s) for measurement noise  [dim_y,N]
% Gr:          gain matrix of planning noise           [dim_x,dim_x,N]
% u:           Input for t=(0:N-1)'*dt [N,dim_u]
%                For dim_u==1, u may also be passed as row vector
% y:           observation                             [N,dim_y]
%                For dim_y==1, y may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:       Covariance matrix [dim_x,dim_x], or time variant covariance matrix [dim_x,dim_x,N) of the process noise 
% Vry:       Covariance matrix [dim_y,dim_y], or time variant covariance matrix [dim_y,dim_y,N) of the measurement noise 
%
% opts:        optional structure with the following fields:
%                do_smoothing: If true, the algorithm performs the backward iterative kalman smoothing as
%                              described in 
%                                     Rauch, H.E., Striebel, C., & Tung, F. (1965). 
%                                     Maximum likelihood estimates of linear dynamic systems.
%                                     AIAA journal, 3 (8), 1445-1450.
%                              In that case x, V, and V_np1_n a conditioned expectation value
%                              across all smoothed x-estimates for the given system parameters.
%                              (default: true)
%
% Returned values:
% x       : The sequence of state estimates                                    [N,dim_x]
% V       : The sequence of covariance matrices of the state-estimation errors [N,dim_x*dim_x]
%             Use reshape(V(n,:),dim_x,dim_x) to read the matrix for index n from V.
% V_np1_n : V_np1_n(n,:) contains the covariance matrix of the estimation errors
%             between the states n+1 and n, i.e. cov(x(n+1)-x_true(n+1),x(n)-x_true(n)). V_np1_n(N,:) contains NaN(1,dim_x*dim_x).
%             Dimension: [N,dim_x*dim_x]. Use reshape(V_np1_n(n,:),dim_x,dim_x) to read the matrix for index n from V_np1_n.
%             For opts.do_smoothing==true, both x, V an V_np1_n return the output of the smoother, otherwise the output
%             of the Kalman forward iteration. 
%         opts.do_smoothing==false: x(n,:), V(n,:) return the aposteriori estimated of state and covariance matrices accounting for measurements y(1:n,:). 
%                                   V_np1_n(n,:) returns the estimates accounting for measurements y(1:n)
%         opts.do_smoothing==true : x, V, and V_np1_n return the smoothed versions of states and covariance matrices accounting for all measurements y(1:N,:).
%
% K       : The sequence of kalman gains [N,dim_x*dim_y]
%             Use reshape(K(n,:),dim_x,dim_y) to read a single matrix from K
% xK,VK   : state and covariance sequences returned by the Kalman forward iteration.
% VK_np1_n           For opts.do_smoothing==false xK, VK, and VK_np1_n are identical with x,V, and V_np1_n, respectively.
% x_ap,V_ap: apriori estimates of the states and covariance matrices of the standard Kalman forward iteration.
%
% copyright: T. Eggert  (2018)
%            Ludwig-Maximilians Universität, München
%            Department of Neurology
%            Feodor-Lynen-Strasse. 19
%            81377 Munich, Germany
%            e-mail: eggert@lrz.uni-muenchen.de

x0=x0(:);
dim_y=size(C,1);
dim_u=size(B,2);

if dim_y~=dim_u
   error('size(B,2) ~= size(C,1)!');
end

if size(y,1)==1 && dim_y==1
   y=y';
end
if size(u,1)==1 && dim_u==1
   u=u';
end

N=size(y,1);
dim_x=size(A,1);

if length(size(A))==2
   A=reshape(repmat(A,1,N),[size(A),N]);
end
if length(size(B))==2
   B=reshape(repmat(B,1,N),[size(B),N]);
end
if length(size(C))==2
   C=reshape(repmat(C,1,N),[size(C),N]);
end
if size(Gv,1)==N && size(Gv,2)==dim_y
   Gv=Gv';
end
if size(Gv,2)==1
   Gv=repmat(Gv,1,N);
end
if length(size(Gr))==2
   Gr=reshape(repmat(Gr,1,N),[size(Gr),N]);
end
if length(size(Vry))==2
   Vry=reshape(repmat(Vry,1,N),[size(Vry),N]);
end
if length(size(Vrx))==2
   Vrx=reshape(repmat(Vrx,1,N),[size(Vrx),N]);
end


default_opts.TrialType=zeros(N,1);
default_opts.do_smoothing=true;

if nargin<12 || isempty(opts)
   opts=[];
end
opts=set_default_parameters(opts,default_opts);



K=NaN(N,dim_x*dim_y);    % kalman matrices
if opts.do_smoothing || nargout>7
   V_ap=zeros(N,dim_x*dim_x);  % time update of the state-error covariance (before measurement)
   x_ap=zeros(N,dim_x);        % state estimates (before measurement)
end
V=zeros(N,dim_x*dim_x);      % covariances of the state estimation-error (after measurement)
x=zeros(N,dim_x);            % state estimates (after measurement)
V_np1_n=NaN(N,dim_x*dim_x);  % covariances of the state estimation-errors cov(x(n+1)-x_true(n+1),x(n)-x_true(n))

Vxbar = Vx0; 
xbar=x0;


for k=1:N  % index 1 corresponds to time0
   if opts.do_smoothing || nargout>7  
      V_ap(k,:)=Vxbar(:)';  % save the apriori estimates needed for smoothing
      x_ap(k,:)=xbar';
   end
   % measurement update
   ybar=C(:,:,k)*xbar;
   
   
   
   
   % change system parameters according to trial type:
   inp=u(k,:)';
   A_=A(:,:,k);
   B_=B(:,:,k);
   C_=C(:,:,k);
   Vrx_=Gr(:,:,k)*Vrx(:,:,k)*Gr(:,:,k)';
   g=diag(Gv(:,k));
   
   
   if ~any(isnan(y(k,:)))
      Ki=Vxbar*C(:,:,k)'/(C_*Vxbar*C_'+Vry(:,:,k));   % Kalman update
      Vx   = Vxbar-Ki*C_*Vxbar;          % A posteriori covariance
      xhat = xbar + Ki*(y(k,:)'-ybar);  % State estimate
      K(k,:)=Ki(:)'; 
      
      xhat_updated=xhat;
      Vx_updated=Vx;
      Km=B_*g*Vry(:,:,k)/(C_*Vxbar*C_'+Vry(:,:,k));
      cor_updated=Km*(y(k,:)'-ybar);
      P_1=Km*C_*Vxbar*A_';
      P_2=Km*Vry(:,:,k)*g*B_';
  
   else  % no observations available at this sampling time
      xhat = xbar;                        % Copy a priori state estimate
      Vx   = Vxbar;                       % Copy a priori covariance factor
   end
   
   
   
   
   % --- Time update (a'priori update) of state and covariance ---
   
   xhat_updated=A_*xhat_updated+B_*inp;
   Vx_updated=A_*Vx_updated*A_' + Vrx_ + B_*g*Vry(:,:,k)*g*B_';
   if any(isnan(y(k,:)))
      cor_updated=A_*cor_updated;
      P_1=A_*P_1*A_';
      P_2=A_*P_2*A_';
   end
   xbar=xhat_updated+cor_updated; % State update
   Vxbar = Vx_updated ...
      -P_1-P_1'-P_2;  % Covariance update

   Vx_np1_p=A_*Vx;
      

   % --- Store results ---
   x(k,:)=xhat';
   V(k,:)= Vx(:)';
   V_np1_n(k,:)=Vx_np1_p(:)';
end

if nargout>4
   xK=x;
end
if nargout>5
   VK=V;
end
if nargout>6
   VK_np1_n=V_np1_n;
end

if opts.do_smoothing
   % run the kalman smoother
   for k=N-1:-1:1
      Vx_t_t=reshape(V(k,:),dim_x,dim_x);
      Vx_tp1_t=reshape(V_ap(k+1,:),dim_x,dim_x);
      J=Vx_t_t*A(:,:,k)'/Vx_tp1_t;
      x(k,:)=x(k,:)+(x(k+1,:)-x_ap(k+1,:))*J';
      Vx=Vx_t_t+J*(reshape(V(k+1,:),dim_x,dim_x)-Vx_tp1_t)*J';
      V(k,:)=Vx(:)';
      V_np1_n_k=reshape(V(k+1,:),dim_x,dim_x)*J';
      V_np1_n(k,:)=V_np1_n_k(:)';
   end
end

end


function [NLL,w_ssqerr,logDet]=incomplete_negLogLike_fun_old(A,B,C,Gv,Gr,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)

if size(y,1)==1
   y=y';
end
[NTrials,NS]=size(y);
if size(inp,1)==1
   inp=inp';
end
NS=max([NS,size(inp,2)]);
if size(opts.TrialType,1)==1
   opts.TrialType=opts.TrialType';
end
NS=max([NS,size(opts.TrialType,2)]);



if size(y,2)<NS
   y=[y,repmat(y(:,end),1,NS-size(y,2))];
end
if size(inp,2)<NS
   inp=[inp,repmat(inp(:,end),1,NS-size(inp,2))];
end
if size(opts.TrialType,2)<NS
   opts.TrialType=[opts.TrialType,repmat(opts.TrialType(:,end),1,NS-size(opts.TrialType,2))];
end

MCV_opts=opts;

NLL=0;
w_ssqerr=0;
logDet=0;
for dat_i=1:size(y,2)
   if dat_i==1 || max(abs(inp(:,dat_i)-inp(:,dat_i-1)))>0 ...
         || max(abs(opts.TrialType(:,dat_i)-opts.TrialType(:,dat_i-1)))>0
      new_input=true;
      MCV_opts.TrialType=opts.TrialType(:,dat_i);
      [x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp(:,dat_i),x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,MCV_opts);
      N=size(y,1);
   else
      new_input=false;
   end
   isvalid=~isnan(y(:,dat_i));
   if new_input || any(isvalid~=last_isvalid)
      [V,L]=eig(VyM(isvalid,isvalid));
      L=diag(L);
      VyMI=V*diag(L.^-1)*V';
   end
   w_ssqerr_i=(y(isvalid,dat_i)-y_hat(isvalid))'*VyMI*(y(isvalid,dat_i)-y_hat(isvalid));   
   NLL=NLL+ (w_ssqerr_i +  sum(log(L))+(N-sum(~isvalid))*log(2*pi)  )/2;
   w_ssqerr=w_ssqerr+w_ssqerr_i;
   logDet=logDet+sum(log(L));
   last_isvalid=isvalid;
end
end


function [incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,Gv,Gr,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%  [incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,Gv,Gr,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%
%     x(n+1)=      A(n)*x(n)+B(n)*(inp(n)+Gv(n)*v)+Gr(n)*r
%
%  For all trial types:
%                       y(n)=C(n)*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)        : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C(n-1)*x_sim(n-1)+Gv(n-1)*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + Gv(n-1)*v(n-1)  for error clamp trials at n-1.
%                              VyType=Vy                            for closed-loop
%                                    =VryEC if ~isnan(VryEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,Vx0)   : initial value of the states
%
% Arguments:
% A:           System matrix                           [dim_x,dim_x,N]
% B:           Input gain matrix                       [dim_x,dim_u,N]
% C:           Output gain matrix                      [dim_y,dim_x,N]
% Gv:          transfer gain(s) for measurement noise  [dim_y,N]
% Gr:          gain matrix of planning noise           [dim_x,dim_x,N]
% y:           Output observations [N,dim_y]
%                For dim_y==1, y may also be passed row vector
% inp:         Input for t=(0:N-1)'*dt [N,dim_u]
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation of the state noise for each state [dim_x,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% VryEC:       Covariance matrix of the measurement noise for error clamp trials [dim_y,dim_y]
%
% opts:        optional structure with the following fields:
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                              (default: zeros(N,1)  )
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
%
%Returned value:
%  incomplete_negLogLike: the incomplete negative log-likelihood
%               w_ssqerr: weighted sum of squared error  (sum{i=1:N}( (y(i)-y_prior(i))'*SIGMA^-1*(y(i)-y_prior(i))   )  )
%
% copyright: T. Eggert  (2020)
%            Ludwig-Maximilians Universität, München
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried,Germany
%            e-mail: eggert@lrz.uni-muenchen.de


opts.do_smoothing=false;
[x_n,Vx_n,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts);
[x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,Gv,Gr,inp,y,x0,Vx0,Vrx_n,Vry_n,opts);

N=size(x,1);
%*** we have to repeat some of the default parameter setting
default_opts.isMissing=false(N,1);
if nargin<12 || isempty(opts)
   opts=[];
end
opts=set_default_parameters(opts,default_opts);

opts.isMissing=opts.isMissing(:);
if length(opts.isMissing)~=N
   error('length(opts.isMissing) must equal the number of observations!');
end
opts.isMissing=(opts.isMissing | any(isnan(y),2));
%****


dim_x=size(A,1);
dim_y=size(C,1);

incomplete_negLogLike=(N-sum(opts.isMissing))*log(2*pi);
w_ssqerr=0;
for k=1:N
   if opts.isMissing(k)
      continue;
   end
   Vk=reshape(V_ap(k,:),dim_x,dim_x);
   SIGMA=C(:,:,k)*Vk*C(:,:,k)'+Vry_n(:,:,k);
   if dim_y>1
      [EV,L]=eig(SIGMA);
      L=diag(L);
      DetSigma=prod(L);
   else
      EV=1;
      L=SIGMA;
      DetSigma=L;
   end
   mu=C(:,:,k)*x_ap(k,:)';
   w_ssqerr_i=(y(k,:)-mu')*(EV*diag(1./L)*EV')*(y(k,:)'-mu);
   incomplete_negLogLike=incomplete_negLogLike + w_ssqerr_i + log(DetSigma);
   w_ssqerr=w_ssqerr+w_ssqerr_i;
end
incomplete_negLogLike=incomplete_negLogLike/2;

end








function AKF=expected_AKF_moving_window_from_cv(y_hat,cv,maxdelay,HalfWindowWidth,indexRange,method)
% Arguments
%      indexRange: [iStart, iEnd] index range from were the AKF is estimated
% HalfWindowWidth: half width of the window used to compute the AKF. Except near the boarders, the window is symmetric around the corresponding index
%          method: 'msq_residual':  E{res(n+d)*res(n)}
%                  'var_residual':  E{res(n+d)*res(n)}-E{res(n+d)}*E{res(n)}
%                         'var_y':  E{y(n+d)*y(n)}-E{y(n+d)}*E{y(n)}
% Return:
%  AKF:  dim=[diff(indexRange)+1,maxdelay+1] each row contains the expected AKF for the corresponding index



if nargin<5 || isempty(method)
   method='msq_residual';
end
method=strmatch_array(method,{'msq_residual','var_residual','var_y'},false);
if isnan(method)
   error('invalid method %s',method);
end


if maxdelay>HalfWindowWidth
   error('HalfWindowWidth must be larger or equal than maxdelay!');
end



L=diff(indexRange)+1;
AKF=NaN(L,maxdelay+1);
for k=1:L
   iR=k+[-1 1]*HalfWindowWidth;
   iR=[max([iR(1),1]),min([iR(2),L])];
   AKF(k,:)=expectedAKF_from_trueCovMatrix(maxdelay,indexRange(1)+iR-1,y_hat,cv,method)';  % identical result, but much faster!!!
end

end


function AKF=expectedAKF_from_trueCovMatrix(maxdelay,indexRange,y_hat,cv,method)
% compute the expected autocovariance for delay of up to maxdelay of the model
%   defined by the mean y_hat and the full covariance matrix cv
%
%   maxdelay: maximum delay
% indexRange: [iStart, iEnd] index range from were the AKF is estimated
%      y_hat: expected y
%         cv: covariance matrix of y
%     method: 1:  E{res(n+d)*res(n)}
%             2:  E{res(n+d)*res(n)}-E{res(n+d)}*E{res(n)}
%             3:  E{y(n+d)*y(n)}-E{y(n+d)}*E{y(n)}
%
% Return:
% AKF: columnvector containing the AKF for delay [0 ... maxdelay] dim=maxdelay+1

cv=cv(indexRange(1):indexRange(2),indexRange(1):indexRange(2));
N=size(cv,1);

if ~isempty(y_hat)
   y_hat=y_hat(indexRange(1):indexRange(2));
else
   y_hat=zeros(N,1);
end

AKF=zeros(1+maxdelay,1);
for delta=0:min(maxdelay,N-1)
   yy_msq=0;
   for i=1:N-delta
      yy_msq=yy_msq+cv(i,i+delta);
      if method>2
         yy_msq=yy_msq+y_hat(i)*y_hat(i+delta); 
      end
   end
   if method>1
      yy_msq=yy_msq/(N-delta-1);
   else
      yy_msq=yy_msq/(N-delta);
   end
   
   mm_sq=0;
   if method>1
      for i=1+delta:N
         for j=1:N-delta
            mm_sq=mm_sq+cv(i,j);         
            if method>2
               mm_sq=mm_sq+y_hat(i)*y_hat(j);  
            end
         end
      end
      mm_sq=mm_sq/(N-delta)/(N-delta-1);
   end
   AKF(delta+1)=yy_msq-mm_sq;
end


end











function [A,B,C,Gv,Gr,inp,x0,Vrx,cva_rx,Vry,cva_ry,VryEC]=mk_TimeVariantSystem(Data,pars)
%        Data.y  (not needed for Data.InputType==0)
%        Data.u
%        Data.TrialType
%        Data.error_clamp  (default: zeros(N,1)   )
%        Data.n_break      (default: ones(N,1)   )
%        Data.InputType    (default: 0)
%  optional fields:
%        Data.StimulusClass         (default: ones(N,1)   )
%        Data.TransferMatrix  (default: diag(ones(NClasses,1))    )
N=size(Data.TrialType,1);
dim_y=1;

if ~isfield(Data,'error_clamp')
   Data.error_clamp=zeros(N,1);
end
if ~isfield(Data,'n_break')
   Data.n_break=ones(N,1);
end
if ~isfield(Data,'InputType')
   Data.InputType=0;
end
if ~isfield(Data,'StimulusClass')
   Data.StimulusClass=ones(N,1);
   Data.TransferMatrix=1;
end
NClasses=length(unique(Data.StimulusClass));
if ~isfield(Data,'TransferMatrix')
   Data.TransferMatrix=diag(ones(NClasses,1));
end
if any(size(Data.TransferMatrix)~=NClasses)
   error('invalid size(Data.TransferMatrix)!');
end

Vry=pars.Vry;
cva_ry=pars.cva_ry;
VryEC=pars.VryEC;

if pars.bf==0 && pars.Vrx11==0 && pars.Vrx12==0
   dim_x_=1;
   Vrx=pars.Vrx22;
   
   x0=pars.xs_start;
   cva_rx=pars.cva_xs;
   b_0=pars.bs;
   c_0=1;
   a_0=pars.as;
else
   dim_x_=2;
   Vrx=[pars.Vrx11,pars.Vrx12;pars.Vrx12,pars.Vrx22];
   
   x0=[pars.xf_start;pars.xs_start];
   cva_rx=[pars.cva_xf;pars.cva_xs];
   b_0=[pars.bf;pars.bs];
   c_0=[1,1];
   a_0=[pars.af;pars.as];

end
dim_x=dim_x_*NClasses;
   
if Data.InputType==1
   A_CL=diag(a_0);
   inp=Data.u-Data.y;
   ia=(1:N)';
   iv=~isnan(inp);
   inp=interp1(ia(iv),inp(iv),ia,'linear','extrap');
   A_EC=A_CL;
   Gv=reshape(repmat(zeros(dim_y,1),1,N),[dim_y,N]);
else
   inp=Data.u;
   A_CL=diag(a_0)-b_0*c_0;
   A_EC=diag(a_0);
   Gv=reshape(repmat(pars.vgain_CL,1,N),[1,N]);
end
a_break=a_0.^pars.abreak_exponent;

Gr=reshape(repmat(diag(ones(dim_x,1)),1,N),[dim_x,dim_x,N]);
if NClasses==1 && Data.TransferMatrix(1,1)==1  %     single class adaptation
   A=reshape(repmat(A_CL,1,N),[size(A_CL),N]);
   B=reshape(repmat(b_0,1,N),[size(b_0),N]);
   C=reshape(repmat(c_0,1,N),[size(c_0),N]);
   i_EC=find(any(repmat(Data.TrialType,1,2)==repmat([1 3],N,1),2));
   if ~isempty(i_EC) 
      for k=1:length(i_EC)
         if Data.InputType==0
            Gv(:,i_EC(k))=pars.vgain_EC;
            A(:,:,i_EC(k))=A_EC;
         end
         inp(i_EC(k),:)=Data.error_clamp(i_EC(k),:);
      end
   end
   i_Break=find(Data.TrialType>1);
   if ~isempty(i_Break)
      for k=1:length(i_Break)
         a_break_n=a_break.^Data.n_break(i_Break(k));
         A(:,:,i_Break(k))=diag(a_break_n)*A(:,:,i_Break(k));
         B(:,:,i_Break(k))=diag(a_break_n)*B(:,:,i_Break(k));
         Gr(:,:,i_Break(k))=diag(a_break_n);
      end
   end
else  % multi class adaptation
   Vrx_new=zeros(dim_x,dim_x);
   cva_rx_new=zeros(dim_x,1);
   x0_new=zeros(dim_x,1);
   for j=1:NClasses
      Vrx_new((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_))=Vrx;
      cva_rx_new((j-1)*dim_x_+(1:dim_x_),1)=cva_rx;
      x0_new((j-1)*dim_x_+(1:dim_x_),1)=x0;
   end
   cva_rx=cva_rx_new;
   Vrx=Vrx_new;
   x0=x0_new;
   clear Vrx_new cva_rx_new x0_new
   
   
   A=zeros(dim_x,dim_x,N);
   B=zeros(dim_x,dim_y,N);
   C=zeros(dim_y,dim_x,N);
   i_CL=find(any(repmat(Data.TrialType,1,2)==repmat([0 2],N,1),2));
   for k=1:length(i_CL)
      for j=1:NClasses
         if Data.StimulusClass(i_CL(k))==j
            if Data.InputType==0
               A((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_),i_CL(k))=diag(a_0)-Data.TransferMatrix(j,j)*b_0*c_0;
            else
               A((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_),i_CL(k))=diag(a_0);
            end
            C(:,(j-1)*dim_x_+(1:dim_x_),i_CL(k))=c_0;
         else
            if Data.InputType==0
               A((j-1)*dim_x_+(1:dim_x_),(Data.StimulusClass(i_CL(k))-1)*dim_x_+(1:dim_x_),i_CL(k))=-Data.TransferMatrix(j,Data.StimulusClass(i_CL(k)))*b_0*c_0;
            end
            A((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_),i_CL(k))=diag(a_0);
         end
         B((j-1)*dim_x_+(1:dim_x_),:,i_CL(k))=Data.TransferMatrix(j,Data.StimulusClass(i_CL(k)))*b_0;
      end
   end
   
   i_EC=find(any(repmat(Data.TrialType,1,2)==repmat([1 3],N,1),2));
   for k=1:length(i_EC)
      if Data.InputType==0
         Gv(:,i_EC(k))=pars.vgain_EC;
      end
      inp(i_EC(k),:)=Data.error_clamp(i_EC(k),:);
      for j=1:NClasses
         if Data.StimulusClass(i_CL(k))==j
             C(:,(j-1)*dim_x_+(1:dim_x_),i_EC(k))=c_0;
         end
         A((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_),i_EC(k))=A_EC;
         B((j-1)*dim_x_+(1:dim_x_),:,i_CL(k))=Data.TransferMatrix(j,Data.StimulusClass(i_EC(k)))*b_0;
      end
   end
   i_Break=find(Data.TrialType>1);
   if ~isempty(i_Break)
      for k=1:length(i_Break)
         a_break_n=a_break.^Data.n_break(i_Break(k));
         for j=1:NClasses
            A((j-1)*dim_x_+(1:dim_x_),:,i_Break(k))=diag(a_break_n)*A((j-1)*dim_x_+(1:dim_x_),:,i_Break(k));
            B((j-1)*dim_x_+(1:dim_x_),:,i_Break(k))=diag(a_break_n)*B((j-1)*dim_x_+(1:dim_x_),:,i_Break(k));
            Gr((j-1)*dim_x_+(1:dim_x_),(j-1)*dim_x_+(1:dim_x_),i_Break(k))=diag(a_break_n);
         end
      end
   end
   
end

end

function [NLL,x_sim,y_sim]=test_loglikelihood_time_variant_system(x_sim,y_sim)
%%

rng(9436597);
c=[1,1];
opts.TrialType=[zeros(150,1);ones(50,1)];
N=length(opts.TrialType);
opts.error_clamp=zeros(N,1);
opts.n_break=ones(N,1);

u=[zeros(20,1);ones(120,1);zeros(60,1)];

opts.TrialType(30)=3;



pars.xf_start=0;
pars.xs_start=0;
pars.af=0.6093;
pars.as=0.9738;
pars.bf=0.3398;
pars.bf=0;
pars.bs=0.0650;
pars.abreak_exponent=5;

pars.vgain_CL=-1;
pars.vgain_EC=0;
pars.Vry=0.002;   
pars.Vrx11=0;   
pars.Vrx12=0;   
pars.Vrx22=0.003;   
pars.cva_xf=0;
pars.cva_xs=0;
pars.cva_ry=0;
pars.VryEC=NaN;





a=[pars.af;pars.as];
b=[pars.bf;pars.bs];
A_EC=diag(a);
A_CL=diag(a)-b*c;

Data.InputType=0;
Data.TrialType=opts.TrialType;
Data.u=u;

Vrx=[pars.Vrx11,pars.Vrx12;pars.Vrx12,pars.Vrx22];
cva_rx=[pars.cva_xf;pars.cva_xs];

Vx0=compute_asymptotic_start_Vrx(pars,Data,1,A_CL,b,c,Vrx,cva_rx);

% manual setting to test mk_TimeVariantSystem
inp=u;
A=reshape([repmat(A_CL,1,150),repmat(A_EC,1,50)],[size(A_CL),N]);
A(:,:,30)=A_EC;
Gv=reshape([repmat(pars.vgain_CL,1,150),repmat(pars.vgain_EC,1,50)],[1,N]);
Gr=reshape(repmat(diag(ones(1,2)),1,N),length(a),length(a),N);
B=reshape(repmat(b,1,N),[size(b),N]);
C=reshape(repmat(c,1,N),[size(c),N]);
x0=[pars.xf_start;pars.xs_start];

i_Break=find(opts.TrialType>1);
for k=1:length(i_Break)
   a_break_n=(a.^pars.abreak_exponent).^opts.n_break(i_Break(k));
   A(:,:,i_Break(k))=diag(a_break_n)*A(:,:,i_Break(k));
   B(:,:,i_Break(k))=diag(a_break_n)*B(:,:,i_Break(k));
   Gr(:,:,i_Break(k))=diag(a_break_n)*Gr(:,:,i_Break(k));
end
i_EC=find(any(repmat(opts.TrialType,1,2)==repmat([1 3],N,1),2));
inp(i_EC,:)=opts.error_clamp(i_EC,:);
Gv(:,i_EC)=pars.vgain_EC;

Data.error_clamp=opts.error_clamp;
Data.n_break=opts.n_break;
Data.StimulusClass=[repmat([1;2;3],66,1);1;3];
Data.TransferMatrix=[1,0.8,0.1;0.8,1,0.8;0.1,0.9,1];

if false
   [A_,B_,C_,Gv_,Gr_,inp_,x0_,Vrx_,cva_rx_,Vry,cva_ry,VryEC]=mk_TimeVariantSystem(Data,pars);
   for k=1:N
      if any(any(abs(A_(:,:,k)-A(:,:,k))>1e-15))
         error('A(:,:,%d)\n',k);
      end
      if any(any(abs(B_(:,:,k)-B(:,:,k))>1e-15))
         error('B(:,:,%d)\n',k);
      end
      if any(any(abs(C_(:,:,k)-C(:,:,k))>1e-15))
         error('C(:,:,%d)\n',k);
      end
      if any(abs(Gv_(:,k)-Gv(:,k))>1e-15)
         error('Gv(:,:,%d)\n',k);
      end
      if any(any(abs(Gr_(:,:,k)-Gr(:,:,k))>1e-15))
         error('Gr(:,:,%d)\n',k);
      end
      if any(any(abs(inp_(k,:)-inp(k,:))>1e-15))
         error('inp(%d,:)\n',k);
      end
   end
else
   [A,B,C,Gv,Gr,inp,x0,Vrx,cva_rx,Vry,cva_ry,VryEC]=mk_TimeVariantSystem(Data,pars);
   Vx0=compute_asymptotic_start_Vrx(pars,Data,1,A(:,:,1),B(:,:,1),C(:,:,1),Vrx,cva_rx);
end

opts.TrialType(opts.TrialType==2)=0;
opts.TrialType(opts.TrialType==3)=1;
if nargin<1
   [x_sim,y_sim]=sim_noisy_system(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,pars.VryEC,2,opts);
end
[NLL,w_ssqerr,logDet]=incomplete_negLogLike_fun_old(A,B,C,Gv,Gr,y_sim(:,1),inp,x0,Vx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,pars.VryEC,opts);
[incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,Gv,Gr,y_sim(:,1),inp,x0,Vx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,pars.VryEC,opts);
fprintf('likelihood error=%10.4e\n',NLL-incomplete_negLogLike);


[x_hat,Vx,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,Gv,Gr,u,x0,Vx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,pars.VryEC,opts);

Vy=zeros(N,1);
for k=1:N,
   Vy(k)=Vry_n(:,:,k)+C(:,:,k)*Vx(:,:,k)*C(:,:,k)';
end

t=(0:N-1)';
figure(1);
clf
hold on
plot(t,x_sim(1:N,2),'-m');
plot(t,x_sim(1:N,1),'-b');
plot(t,y_sim,'-r');



figure(2);
clf
hold on
plot(t,Vy,'-b');
end


