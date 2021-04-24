% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
% Library for computing the covariance matrix of the output of a closed-loop time-discrete linear system and its likelihood
function dls_lib=DiscreteLinearSystem_lib()

% [x_sim,y_sim,r,v]=sim_noisy_system(A,B,C,inp,x0,V0,Vx,cva_rx,Vy,cva_ry,VyEC,Nsim,opts)
dls_lib.sim_noisy_system=@sim_noisy_system;  
% [x_hat,Vx,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.expected_DiscreteSystem_stats=@ expected_DiscreteSystem_stats;
%[x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.MeanCovMatrix_DiscretLinearSystem=@MeanCovMatrix_DiscretLinearSystem;
%[L,Vxx_Asymp]=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vgain,u_inf,Vrx,Vry)
dls_lib.analyse_asymptotic_state_cov=@analyse_asymptotic_state_cov;
%[VyAsymp,Vxx_Asymp]=compute_asymptotic_Autocov_y(u_inf,A,B,C,vg,Vrx,cva_rx,Vry,maxdelay,cva_ry,isErrorClamp)
dls_lib.compute_asymptotic_Autocov_y=@compute_asymptotic_Autocov_y;
%Vrx0=compute_asymptotic_start_Vrx(pars,Data,ci,A,B,C,Vrx,cva_rx)
dls_lib.compute_asymptotic_start_Vrx=@compute_asymptotic_start_Vrx;
%[x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,u,y,x0,V0,var_x,var_y,opts)
dls_lib.kalman_observer=@kalman_observer;
%[NLL,w_ssqerr,logDet]=incomplete_negLogLike_fun_old(A,B,C,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.incomplete_negLogLike_fun_old=@incomplete_negLogLike_fun_old;
%[incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,y,u,x0,V0,Vx,cva_rx,Vry,cva_ry,VryEC,opts)
dls_lib.incomplete_negLogLike_fun_new=@incomplete_negLogLike_fun_new;
end


function [x_sim,y_sim,r,v]=sim_noisy_system(A,B,C,inp,x0,V0,Vx,cva_rx,Vy,cva_ry,VyEC,Nsim,opts)
%x_sim=sim_noisy_system(A,B,C,u,x0,V0,Vx,Vy,Nsim,opts)
%
% For opts.InputType==0: (driven by input inp=u)
%  closed loop trials:          x(n+1)=      A*x(n)+B*(inp(n)+vgain.CL*v)+r                                   for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=(A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r                        for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(       A*x(n)+B*(inp(n)+vgain.CL*v)+r  )            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*( (A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r  ) for 2<opts.TrialType(n)
%
% For opts.InputType==1: (driven by error inp=u-y_observed)
%  closed loop trials:          x(n+1)=A*x(n)+B*inp(n)+r                                for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=A*x(n)+B*error_clamp(k,:)'+r                     for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*inp(n)+r)            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*error_clamp(k,:)'+r) for 2<opts.TrialType(n)
%
%  For all trial types:
%                       y(n)=C*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)        : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C*x_sim(n-1)+vgain.CL*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + vgain.EC*v(n-1)  for error clamp trials at n-1.
%                              cva_ry~=0 is NOT ALLOWED for InputType==1
%                              VyType=Vy                            for closed-loop
%                                    =VyEC if ~isnan(VyEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,V0)   : initial value of the states
%
% Arguments:
% A:           System matrix         [dim_x,dim_x]
% B:           Input gain matrix     [dim_x,dim_u]
% C:           Output gain matrix    [dim_y,dim_x]
% inp:         Input for t=(1:N)'*dt [N,dim_u]
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% V0:          The initial state covariance matrix. [dim_x,dim_x]
% Vx:          Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation of the state noise for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vy:          Covariance matrix of the measurement noise [dim_y,dim_y]
% VyEC:        if ~isnan(VyEC(1,1)): Covariance matrix of the measurement noise for error-clamp trials [dim_y,dim_y]
%              default: NaN
% opts:        optional structure with the following fields:
%                 InputType:   Logical scalar specifying the input type:
%                                 0: driven by input u
%                                 1: driven by observed error u-y 
%                              (default: 0)
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                                 1<TrialType(n)<=2: closed loop trial n is followed by a set break
%                                 2<TrialType(n)   : error clamp trial n is followed by a set break
%                              (default: zeros(N,1)  )
%                 Abreak:      The matrix that specifies the state update during the set break as
%                              defined above.
%                              (default: eye(dim_x)  )
%                 vgain:       Structure with fields .CL and .EC, containing the transfer factor of the measurement
%                              measurement noise to the input of the state update for closed loop trials and error
%                              clamp trials, respectively. A value of -1 indicates that the measurement noise is transfered 
%                              completely to the state update (default: vgain.CL=0; vgain.EC=0) 
%                              vgain is irrelevant for InputType~=0
%                 error_clamp: Numerical matrix with dimension [N,dim_u] which specifies state update
%                              in the error clamp trials as defined above. For dim_y==dim_u==1, error_clamp
%                              may also be passed as row vector.
%                              (default: zeros(N,dim_u)  )
%                     n_break: break lenght (default: ones(N,1))
%                noise_method: ==0: covariance matrix of signal dependent noise is  diag(cva_rx.^2.*x_sim(n).^2)
%                              ~=0:                     "                           diag(cva_rx.^2.*(diag(Vx_expected(n))+x_expected(n).^2));
%                                   (default=1)
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
% x_sim       : The simulated state sequence for t=(0:N)'*dt           [N+1,Nsim*dim_x]   
% y_sim       : The simulated output for t=(0:N)'*dt                   [N+1,Nsim*dim_y]
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

if dim_y~=dim_u,
   error('size(B,2) ~= size(C,1)!');
end;

if size(inp,1)==1 && dim_u==1,
   inp=inp';
end;

N=size(inp,1);
dim_x=size(A,1);

if nargin<11 || isempty(VyEC),
   VyEC=NaN;
end;
if isnan(VyEC(1,1)),
   VyEC=Vy;
end;

if nargin<12 || isempty(Nsim),
   Nsim=100;
end;
cva_rx=abs(cva_rx(:));

default_opts.InputType=0;
default_opts.TrialType=zeros(N,1);
default_opts.Abreak=eye(dim_x);
default_opts.error_clamp=zeros(N,dim_u);
default_opts.n_break=ones(N,dim_u);
default_opts.vgain.CL=zeros(dim_u,1);
default_opts.vgain.EC=zeros(dim_u,1);

default_opts.noise_method=1;
default_opts.signed_EDPMN=false;



if nargin<13 || isempty(opts),
   opts=[];
end;
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>3),
   error('TrialType may not exceed 3!');
end;
if any(opts.TrialType<0),
   error('TrialType must exceed -1!');
end;

opts.vgain.CL=opts.vgain.CL(:);
opts.vgain.EC=opts.vgain.EC(:);

if ~any(opts.InputType==[0 1]),
   error('invalid input type!');
end;

if size(opts.error_clamp,1)==1 && dim_y==1 && dim_u==1,
   opts.error_clamp=opts.error_clamp';
end;

if length(opts.TrialType)~=N,
   error('length(opts.TrialType) must equal the number of observations!');
end;


if any(opts.TrialType==1 | opts.TrialType==3),
   if opts.InputType==0,
      AErrorClamp=A+B*C;
   else
      AErrorClamp=A;
   end;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error-clamp trials!');
   end;
end;

if any(opts.TrialType==2),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break=opts.Abreak*A;
   B_break=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if any(opts.TrialType==3),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break_EC=opts.Abreak*AErrorClamp;
   B_break_EC=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if opts.InputType==1,
   opts.vgain.CL=zeros(dim_u,1);
   opts.vgain.EC=zeros(dim_u,1);
   if any(cva_ry>0),
      error('can''t simulate error dependent motor noise with InputType==1!');
   end;
end;

x_sim=zeros(N+1,Nsim*dim_x);  % t=(0:N)'*dt;
y_sim=zeros(N+1,Nsim*dim_y);  

opts.TrialType(N+1)=opts.TrialType(N);  % this is only to allow type dependent Vry at t=N

%** compute the expected state, the expected state covariance matrix, and the covariance matrix of the state noise

if all(cva_rx==0) && all(cva_ry==0),
   Vrx_n=reshape(repmat(Vx,1,N+1),dim_x,dim_x,N+1);
   Vry_n=reshape(repmat(Vy,1,N+1),dim_y,dim_y,N+1);
   is_EC=any(repmat(opts.TrialType,1,2)==repmat([1 3],N+1,1),2);
   if any(is_EC),
      N_EC=sum(is_EC);
      Vry_n(:,:,is_EC)=repmat(VyEC,1,1,N_EC);
   end;
elseif opts.noise_method==0,
   Vrx_n=zeros(dim_x,dim_x,N+1);
   Vry_n=zeros(dim_y,dim_y,N+1);
else
   Vrx_n=zeros(dim_x,dim_x,N+1);
   Vrx_n(:,:,1)=Vx+diag(cva_rx.^2.*(diag(V0)+x0.^2));
   
   Vry_n=zeros(dim_y,dim_y,N+1);
   if any(opts.TrialType(1)==[1 3]),
      Vry_n(:,:,1)=VyEC;
   else
      Vry_n(:,:,1)=Vy;
   end;
   
   x_expected=x0;
   Vx_expected=V0;
   for k=2:N+1,
      
      if any(opts.TrialType(k)==[0 2]),  % closed loop trial
         VyType=Vy;
      else
         VyType=VyEC;
      end;
      if any(cva_ry>0),
         % update the motor noise covariance matrix
         if any(opts.TrialType(k-1)==[0 2]),  % closed loop trial
            err_nm1_expected=inp(k-1,:)'-C*x_expected;
            if opts.signed_EDPMN,
               expected_err_sq=zeros(size(inp,2));
               cv_expected_cx=C*Vx_expected*C';
               cv_expected_err=cv_expected_cx+diag(opts.vgain.CL)*Vry_n(:,:,k-1)*diag(opts.vgain.CL);
               for inp_i=1:size(inp,2),
                  m_cx_err=[C(inp_i,:)*x_expected;err_nm1_expected(inp_i)];
                  S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
                  expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
               end;
            else
               expected_err_sq=C*Vx_expected*C'+ err_nm1_expected*err_nm1_expected'+diag(opts.vgain.CL)*Vry_n(:,:,k-1)*diag(opts.vgain.CL);
            end;
         else
            if opts.signed_EDPMN,
               expected_err_sq=zeros(size(inp,2));
               cv_expected_cx=C*Vx_expected*C';
               cv_expected_err=diag(opts.vgain.EC)*Vry_n(:,:,k-1)*diag(opts.vgain.EC);
               for inp_i=1:size(inp,2),
                  m_cx_err=[C(inp_i,:)*x_expected;opts.error_clamp(k-1,inp_i)];
                  S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
                  expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
               end;
            else
               expected_err_sq=opts.error_clamp(k-1,:)'*opts.error_clamp(k-1,:) +diag(opts.vgain.EC)*Vry_n(:,:,k-1)*diag(opts.vgain.EC);
            end;
         end;
         Vry_n(:,:,k)=VyType+diag(cva_ry.^2.*diag(expected_err_sq));
      else
         Vry_n(:,:,k)=VyType;
      end;
      
      
      if opts.TrialType(k-1)==0,      % closed loop trial
         x_expected=A*x_expected+B*inp(k-1,:)';
         Vx_expected=A*Vx_expected*A'+Vrx_n(:,:,k-1) + B*diag(opts.vgain.CL)*Vry_n(:,:,k-1)*diag(opts.vgain.CL)*B';
      elseif opts.TrialType(k-1)==1,  % error clamp trial
         x_expected=AErrorClamp*x_expected+B*opts.error_clamp(k-1,:)';
         Vx_expected=AErrorClamp*Vx_expected*AErrorClamp'+Vrx_n(:,:,k-1) + B*diag(opts.vgain.EC)*Vry_n(:,:,k-1)*diag(opts.vgain.EC)*B';
      elseif opts.TrialType(k-1)==2,  % set break after closed loop trial k-1
         if opts.n_break(k-1)~=1,
            A_break_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*A;
            B_break_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
            Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k-1));
         else
            A_break_n=A_break;
            B_break_n=B_break;
            Abreak_n=opts.Abreak;
         end;
         x_expected=A_break_n*x_expected+B_break_n*inp(k-1,:)';
         Vx_expected=A_break_n*Vx_expected*A_break_n'+Abreak_n*Vrx_n(:,:,k-1)*Abreak_n' + B_break_n*diag(opts.vgain.CL)*Vry_n(:,:,k-1)*diag(opts.vgain.CL)*B_break_n';
      else   % set break after error clamp trial k-1
         if opts.n_break(k-1)~=1,
            A_break_EC_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*AErrorClamp;
            B_break_EC_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
            Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k-1));
         else
            A_break_EC_n=A_break_EC;
            B_break_EC_n=B_break_EC;
            Abreak_n=opts.Abreak;
         end;
         x_expected=A_break_EC_n*x_expected+B_break_EC_n*opts.error_clamp(k-1,:)';
         Vx_expected=A_break_EC_n*Vx_expected*A_break_EC_n'+Abreak_n*Vrx_n(:,:,k-1)*Abreak_n' + B_break_EC_n*diag(opts.vgain.EC)*Vry_n(:,:,k-1)*diag(opts.vgain.EC)*B_break_EC_n';
      end;
      
        %** update the state covariance matrix
      if any(cva_rx>0),
         Vrx_n(:,:,k)=Vx+diag(cva_rx.^2.*(diag(Vx_expected)+x_expected.^2));
      else
         Vrx_n(:,:,k)=Vx;
      end;
   end;
      
   
end;


for si=1:Nsim,
   if all(cva_ry==0),
      v=mvnrnd(zeros(1,dim_y),Vy,N+1);
      is_EC=any(repmat(opts.TrialType,1,2)==repmat([1 3],N+1,1),2);
      if any(is_EC),
         N_EC=sum(is_EC);
         v(is_EC,:)=mvnrnd(zeros(1,dim_y),VyEC,N_EC);
      end;
   else
      v=zeros(N,dim_y);
      if opts.noise_method==0,
         if any(opts.TrialType(1)==[1 3]),
            Vry_n(:,:,1)=VyEC;
         else
            Vry_n(:,:,1)=Vy;
         end;
      end;
      v(1,:)=mvnrnd(zeros(1,dim_y),Vry_n(:,:,1),1);
   end;
   
   xbar=x0+mvnrnd(zeros(1,dim_x),V0,1)';
   if all(cva_rx==0),
      r=mvnrnd(zeros(1,dim_x),Vx,N);
   else
      r=zeros(N,dim_x);
      if opts.noise_method==0,
         Vrx_n(:,:,1)=Vx+diag(cva_rx.^2.*xbar.^2);
      end;
      r(1,:)=mvnrnd(zeros(1,dim_x),Vrx_n(:,:,1),1);
   end;
   
   
   x=zeros(N+1,dim_x);
   y=zeros(N+1,dim_y);
   x(1,:)=xbar';
   y(1,:)=(C*xbar+v(1,:)')';
   for k=2:N+1,
      if any(opts.TrialType(k)==[0 2]),  % closed loop trial
         VyType=Vy;
      else
         VyType=VyEC;
      end;
      if opts.TrialType(k-1)==0,      % closed loop trial
         err_nm1=inp(k-1,:)'+diag(opts.vgain.CL)*v(k-1,:)'-C*xbar;
         xbar=A*xbar+B*(inp(k-1,:)'+diag(opts.vgain.CL)*v(k-1,:)')+r(k-1,:)';
      elseif opts.TrialType(k-1)==1,  % error clamp trial
         err_nm1=opts.error_clamp(k-1,:)'+diag(opts.vgain.EC)*v(k-1,:)';
         xbar=AErrorClamp*xbar+B*(opts.error_clamp(k-1,:)'+diag(opts.vgain.EC)*v(k-1,:)')+r(k-1,:)';
      elseif opts.TrialType(k-1)==2,  % set break after closed loop trial k-1
         if opts.n_break(k-1)~=1,
            A_break_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*A;
            B_break_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
            Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k-1));
         else
            A_break_n=A_break;
            B_break_n=B_break;
            Abreak_n=opts.Abreak;
         end;
         err_nm1=inp(k-1,:)'+diag(opts.vgain.CL)*v(k-1,:)'-C*xbar;
         xbar=A_break_n*xbar+B_break_n*(inp(k-1,:)'+diag(opts.vgain.CL)*v(k-1,:)')+Abreak_n*r(k-1,:)';
      else   % set break after error clamp trial k-1
         if opts.n_break(k-1)~=1,
            A_break_EC_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*AErrorClamp;
            B_break_EC_n=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
            Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k-1));
         else
            A_break_EC_n=A_break_EC;
            B_break_EC_n=B_break_EC;
            Abreak_n=opts.Abreak;
         end;
         err_nm1=opts.error_clamp(k-1,:)'+diag(opts.vgain.EC)*v(k-1,:)';
         xbar=A_break_EC_n*xbar+B_break_EC_n*(opts.error_clamp(k-1,:)'+diag(opts.vgain.EC)*v(k-1,:)')+Abreak_n*r(k-1,:)';
      end;
      
      if any(cva_ry>0),
         if opts.noise_method==0,
            if ~opts.signed_EDPMN || err_nm1*(C*x(k-1,:)')>0,
               Vry_n(:,:,k)=VyType+diag(cva_ry.^2.*err_nm1.^2);
            else
               Vry_n(:,:,k)=VyType;
            end
         end;
         v(k,:)=mvnrnd(zeros(1,dim_y),Vry_n(:,:,k),1);
      end;
      
      if any(cva_rx>0),
         if opts.noise_method==0,
            Vrx_n(:,:,k)=Vx+diag(cva_rx.^2.*xbar.^2);
         end;
         r(k,:)=mvnrnd(zeros(1,dim_x),Vrx_n(:,:,k),1);
      end;
      x(k,:)=xbar';
      y(k,:)=(C*xbar+v(k,:)')';
   end;
   x_sim(:,(si-1)*dim_x+(1:dim_x))=x; 
   y_sim(:,(si-1)*dim_y+(1:dim_y))=y; 
end;
end


function [x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%x_sim=MeanCovMatrix_DiscretLinearSystem(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
%
% For opts.InputType==0: (driven by input inp=u)
%  closed loop trials:          x(n+1)=      A*x(n)+B*(inp(n)+vgain.CL*v)+r                                   for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=(A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r                        for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(       A*x(n)+B*(inp(n)+vgain.CL*v)+r  )            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*( (A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r  ) for 2<opts.TrialType(n)
%
% For opts.InputType==1: (driven by error inp=u-y_observed)
%  closed loop trials:          x(n+1)=A*x(n)+B*inp(n)+r                                for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=A*x(n)+B*error_clamp(k,:)'+r                     for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*inp(n)+r)            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*error_clamp(k,:)'+r) for 2<opts.TrialType(n)
%
%  For all trial types:
%                       y(n)=C*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)    : process noise
%                       v(n) ~ N(0,VryType+diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C*x(n-1)+vgain.CL*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + vgain.EC*v(n-1)  for error clamp trials at n-1.
%                              cva_ry~=0 is NOT ALLOWED for InputType==1
%                              VryType=Vry                            for closed-loop
%                                     =VryEC if ~isnan(VryEC(1,11)), otherwise =Vry  for error-clamp
%                       x0~ N(0,Vx0)  : initial value of the states
%
% Arguments:
% A:           System matrix       [dim_x,dim_x]
% B:           Input gain matrix   [dim_x,dim_u]
% C:           Output gain matrix  [dim_y,dim_x]
% inp:         Input               [N,dim_u]            ( t=(0:N-1)'*dt )
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       if ~isnan(VryEC(1,1)): Covariance matrix of the measurement noise for error-clamp trials [dim_y,dim_y]
%              default: NaN
% opts:        optional structure with the following fields:
%                 InputType:   Logical scalar specifying the input type:
%                                 0: driven by input u
%                                 1: driven by observed error u-y 
%                              (default: 0)
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                                 1<TrialType(n)<=2: closed loop trial n is followed by a set break
%                                 2<TrialType(n)   : error clamp trial n is followed by a set break
%                              (default: zeros(N,1)  )
%                 Abreak:      The matrix that specifies the state update during the set break as
%                              defined above.
%                              (default: eye(dim_x)  )
%                 vgain:       Structure with fields .CL and .EC, containing the transfer factor of the measurement
%                              measurement noise to the input of the state update for closed loop trials and error
%                              clamp trials, respectively. A value of -1 indicates that the measurement noise is transfered 
%                              completely to the state update (default: vgain.CL=0; vgain.EC=0) 
%                              vgain is irrelevant for InputType~=0
%                 error_clamp: Numerical matrix with dimension [N,dim_u] which specifies state update
%                              in the error clamp trials as defined above. For dim_y==dim_u==1, error_clamp
%                              may also be passed as row vector.
%                              (default: zeros(N,dim_u)  )
%                     n_break: break lenght (default: ones(N,1))
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
% Returned values:
% x_hat       : The expected state sequence         [N+1,dim_x]      ( t=(0:N)'*dt )
% y_hat       : The expected output                 [N+1,dim_y]
% VyM         : The covariance matrix of the output [(N+1)*dim_y,(N+1)*dim_y]
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

if dim_y~=dim_u,
   error('size(B,2) ~= size(C,1)!');
end;

if size(inp,1)==1 && dim_u==1,
   inp=inp';
end;

N=size(inp,1);
dim_x=size(A,1);

if any(any(isnan(inp))),
   error('no missings are allowed in inp!');
end;

cva_rx=cva_rx(:);
cva_ry=cva_ry(:);
default_opts.InputType=0;
default_opts.TrialType=zeros(N,1);
default_opts.Abreak=eye(dim_x);
default_opts.error_clamp=zeros(N,dim_u);
default_opts.n_break=ones(N,1);
default_opts.vgain.CL=zeros(dim_y,1);
default_opts.vgain.EC=zeros(dim_y,1);
default_opts.signed_EDPMN=false;



if nargin<11 || isempty(VryEC),
   VryEC=NaN;
end;
if isnan(VryEC(1,1)),
   VryEC=Vry;
end;

if nargin<12 || isempty(opts),
   opts=[];
end;
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>3),
   error('TrialType may not exceed 3!');
end;
if any(opts.TrialType<0),
   error('TrialType must exceed -1!');
end;

opts.vgain.CL=opts.vgain.CL(:);
opts.vgain.EC=opts.vgain.EC(:);

if ~any(opts.InputType==[0 1]),
   error('invalid input type!');
end;

if size(opts.error_clamp,1)==1 && dim_y==1 && dim_u==1,
   opts.error_clamp=opts.error_clamp';
end;

if length(opts.TrialType)~=N,
   error('length(opts.TrialType) must equal the number of observations!');
end;



if any(opts.TrialType==1 | opts.TrialType==3),
   if opts.InputType==0,
      AErrorClamp=A+B*C;
   else
      AErrorClamp=A;
   end;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error-clamp trials!');
   end;
end;

if any(opts.TrialType==2),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break=opts.Abreak*A;
   B_break=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if any(opts.TrialType==3),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break_EC=opts.Abreak*AErrorClamp;
   B_break_EC=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if opts.InputType==1,
   vgain=zeros(dim_y,N);
   if any(cva_ry>0),
      error('error dependent motor noise is not compatible with InputType==1!');
   end;
else
   vgain=repmat(opts.vgain.CL,1,N);
   ec_i=any(repmat(opts.TrialType,1,2)==repmat([1 3],N,1),2);
   vgain(:,ec_i)=repmat(opts.vgain.EC,1,sum(ec_i));
end;

Dn=zeros(dim_x,dim_x,N);
An=Dn;
Bn=zeros(dim_x,dim_y,N);
Zn=zeros(dim_y,N);

% compute the expected output and its covariance matrix
y_hat=zeros(N+1,dim_y);      % the expected output
Vx=zeros(dim_x,dim_x,N+1);   % the covariance matrix of the state noise
Vy=zeros(dim_y,dim_y,N+1);   % the variance of the output
Vry_n=zeros(dim_y,dim_y,N+1);% the variance of the motor noise

x_hat=zeros(N+1,dim_x);
x_hat(1,:)=x0';
y_hat(1,:)=x0'*C';
xbar=x0;

opts.TrialType(N+1)=opts.TrialType(N);  % this is only to allow type dependent Vry at t=N
if any(opts.TrialType(1)==[0,2]),
   Vry_n(:,:,1)=Vry;
else
   Vry_n(:,:,1)=VryEC;
end;
Vx(:,:,1)=Vx0;
Vy(:,:,1)=C*Vx(:,:,1)*C'+Vry_n(:,:,1);
for k=2:N+1,
   if any(opts.TrialType(k)==[0 2]),  % closed loop trial
      VryType=Vry;
   else
      VryType=VryEC;
   end;
   if opts.TrialType(k-1)==0,  % closed loop trial
      An(:,:,k-1)=A;
      Bn(:,:,k-1)=B;
      Dn(:,:,k-1)=eye(dim_x);
      Zn(:,k-1)=inp(k-1,:)';
   elseif opts.TrialType(k-1)==1,  % error clamp trial
      An(:,:,k-1)=AErrorClamp;
      Bn(:,:,k-1)=B;
      Dn(:,:,k-1)=eye(dim_x);
      Zn(:,k-1)=opts.error_clamp(k-1,:)';
   elseif opts.TrialType(k-1)==2,  % set break after closed loop trial k
      if opts.n_break(k-1)~=1
         An(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*A;
         Bn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
         Dn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1));
      else
         An(:,:,k-1)=A_break;
         Bn(:,:,k-1)=B_break;
         Dn(:,:,k-1)=opts.Abreak;
      end
      Zn(:,k-1)=inp(k-1,:)';
   elseif opts.TrialType(k-1)==3,  % set break after error clamp trial k
      if opts.n_break(k-1)~=1
         An(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*AErrorClamp;
         Bn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
         Dn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1));
      else
         An(:,:,k-1)=A_break_EC;
         Bn(:,:,k-1)=B_break_EC;
         Dn(:,:,k-1)=opts.Abreak;
      end
      Zn(:,k-1)=opts.error_clamp(k-1,:)';
   end;
   
   
   if any(cva_ry>0),
      
      % update the motor noise covariance matrix
      if any(opts.TrialType(k-1)==[0 2]),  % closed loop trial
         err_nm1_expected=inp(k-1,:)'-C*xbar;
         if opts.signed_EDPMN,
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C*Vx(:,:,k-1)*C';
            cv_expected_err=cv_expected_cx+diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
            for inp_i=1:size(inp,2),
               m_cx_err=[C(inp_i,:)*xbar;err_nm1_expected(inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end;
         else
            expected_err_sq=C*Vx(:,:,k-1)*C'+ err_nm1_expected*err_nm1_expected'+diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
         end;
      else
         if opts.signed_EDPMN,
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C*Vx(:,:,k-1)*C';
            cv_expected_err=diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
            for inp_i=1:size(inp,2),
               m_cx_err=[C(inp_i,:)*xbar;opts.error_clamp(k-1,inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end;
         else
            expected_err_sq=opts.error_clamp(k-1,:)'*opts.error_clamp(k-1,:) +diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
         end;
      end;
      Vry_n(:,:,k)=VryType+diag(cva_ry.^2.*diag(expected_err_sq));
   else
      Vry_n(:,:,k)=VryType;
   end;
   
   
   Vrx_n=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,k-1))+xbar.^2));
   xbar=An(:,:,k-1)*xbar+Bn(:,:,k-1)*Zn(:,k-1);
   Vx_tmp=An(:,:,k-1)*Vx(:,:,k-1)*An(:,:,k-1)'+Dn(:,:,k-1)*Vrx_n*Dn(:,:,k-1)' + Bn(:,:,k-1)*diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1))*Bn(:,:,k-1)';
   %** clip Vx: ****
   Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
   Vx(:,:,k)=Vx_tmp;
   
   
   x_hat(k,:)=xbar';
   y_hat(k,:)=xbar'*C';
   
   Vy(:,:,k)=C*Vx(:,:,k)*C'+Vry_n(:,:,k);
end;



% compute the full covariance matrix of y:
VyM=zeros((N+1)*dim_y, (N+1)*dim_y);
for j=1:N,
   Vij=Vx(:,:,j);
   VyM((j-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))=C*Vij*C'+Vry_n(:,:,j);
   
   if opts.InputType==0,
      v_ij=Bn(:,:,j)*diag(vgain(:,j))*Vry_n(:,:,j);
   else
      v_ij=zeros(dim_x,1);
   end;
      
   for i=j+1:N+1,
      
      Vij=An(:,:,i-1)*Vij;
      
%       COV_X(:,:,i,j)=Vij;  % the cross covariance matrices may be stored for later use
%       COV_X(:,:,j,i)=Vij';
      if i>j+1,
         v_ij=An(:,:,i-1)*v_ij;
      end;
      VyM((i-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))=C*(Vij*C'+v_ij);
      VyM((j-1)*dim_y+(1:dim_y),(i-1)*dim_y+(1:dim_y))=VyM((i-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))';
   end;
end;
j=N+1;
Vij=Vx(:,:,j);
VyM((j-1)*dim_y+(1:dim_y),(j-1)*dim_y+(1:dim_y))=C*Vij*C'+Vry_n(:,:,j);

end


function [x_hat,Vx,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)
% this is a copy of the first loop of MeanCovMatrix_DiscretLinearSystem(A,B,C,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)


% For opts.InputType==0: (driven by input inp=u)
%  closed loop trials:          x(n+1)=      A*x(n)+B*(inp(n)+vgain.CL*v)+r                        for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=(A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r             for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(       A*x(n)+B*(inp(n)+vgain.CL*v)+r  )            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*( (A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r  ) for 2<opts.TrialType(n)
%
% For opts.InputType==1: (driven by error inp=u-y_observed)
%  closed loop trials:          x(n+1)=A*x(n)+B*inp(n)+r                                for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=A*x(n)+B*error_clamp(k,:)'+r                     for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*inp(n)+r)            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*error_clamp(k,:)'+r) for 2<opts.TrialType(n)
%
%  For all trial types:
%                       y(n)=C*x(n) + v
%                       r(n) ~ N(0,Vrx+diag(cva_rx.^2.*x(n).^2)    : process noise
%                       v(n) ~ N(0,VryType+diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C*x(n-1)+vgain.CL*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + vgain.EC*v(n-1)  for error clamp trials at n-1.
%                              cva_ry~=0 is NOT ALLOWED for InputType==1
%                              VryType=Vry                            for closed-loop
%                                     =VryEC if ~isnan(VryEC(1,11)), otherwise =Vry  for error-clamp
%                       x0~ N(0,Vx0)  : initial value of the states
%
% Arguments:
% A:           System matrix       [dim_x,dim_x]
% B:           Input gain matrix   [dim_x,dim_u]
% C:           Output gain matrix  [dim_y,dim_x]
% inp:         Input               [N,dim_u]            ( t=(0:N-1)'*dt )
%                For dim_u==1, inp may also be passed as row vector
% x0:          The initial state. Row- or column vector with length dim_x
% Vx0:         The initial state covariance matrix. [dim_x,dim_x]
% Vrx:         Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       if ~isnan(VryEC(1,1)): Covariance matrix of the measurement noise for error-clamp trials [dim_y,dim_y]
%              default: NaN
% opts:        optional structure with the following fields:
%                 InputType:   Logical scalar specifying the input type:
%                                 0: driven by input u
%                                 1: driven by observed error u-y 
%                              (default: 0)
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial ( t=(0:N-1)'*dt )
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                                 1<TrialType(n)<=2: closed loop trial n is followed by a set break
%                                 2<TrialType(n)   : error clamp trial n is followed by a set break
%                              (default: zeros(N,1)  )
%                 Abreak:      The matrix that specifies the state update during the set break as
%                              defined above.
%                              (default: eye(dim_x)  )
%                 vgain:       Structure with fields .CL and .EC, containing the transfer factor of the measurement
%                              measurement noise to the input of the state update for closed loop trials and error
%                              clamp trials, respectively. A value of -1 indicates that the measurement noise is transfered 
%                              completely to the state update (default: vgain.CL=0; vgain.EC=0) 
%                              vgain is irrelevant for InputType~=0
%                 error_clamp: Numerical matrix with dimension [N,dim_u] which specifies state update
%                              in the error clamp trials as defined above. For dim_y==dim_u==1, error_clamp
%                              may also be passed as row vector.
%                              (default: zeros(N,dim_u)  )
%                     n_break: break lenght (default: ones(N,1))
%                signed_EDPMN: logical scalar,  true: the variance of the error dependent motor noise (EDPM) is zero if the sign of C*x differs from the sign
%                                                     of the error
%                                              false: the variance of the error dependent motor noise v(n) is diag(cva_ry.^2.*err(n-1).^2)
%                                   (default=false);
% Returned values:
% x_hat       : The expected state sequence         [N+1,dim_x]      ( t=(0:N)'*dt )
% y_hat       : The expected output                 [N+1,dim_y]
% VyM         : The covariance matrix of the output [(N+1)*dim_y,(N+1)*dim_y]
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

if dim_y~=dim_u,
   error('size(B,2) ~= size(C,1)!');
end;

if size(inp,1)==1 && dim_u==1,
   inp=inp';
end;

N=size(inp,1);
dim_x=size(A,1);

if any(any(isnan(inp))),
   error('no missings are allowed in inp!');
end;

cva_rx=cva_rx(:);
cva_ry=cva_ry(:);
default_opts.InputType=0;
default_opts.TrialType=zeros(N,1);
default_opts.Abreak=eye(dim_x);
default_opts.error_clamp=zeros(N,dim_u);
default_opts.n_break=ones(N,1);
default_opts.vgain.CL=zeros(dim_y,1);
default_opts.vgain.EC=zeros(dim_y,1);
default_opts.signed_EDPMN=false;



if nargin<11 || isempty(VryEC),
   VryEC=NaN;
end;
if isnan(VryEC(1,1)),
   VryEC=Vry;
end;

if nargin<12 || isempty(opts),
   opts=[];
end;
opts=set_default_parameters(opts,default_opts);

% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>3),
   error('TrialType may not exceed 3!');
end;
if any(opts.TrialType<0),
   error('TrialType must exceed -1!');
end;

opts.vgain.CL=opts.vgain.CL(:);
opts.vgain.EC=opts.vgain.EC(:);

if ~any(opts.InputType==[0 1]),
   error('invalid input type!');
end;

if size(opts.error_clamp,1)==1 && dim_y==1 && dim_u==1,
   opts.error_clamp=opts.error_clamp';
end;

if length(opts.TrialType)~=N,
   error('length(opts.TrialType) must equal the number of observations!');
end;



if any(opts.TrialType==1 | opts.TrialType==3),
   if opts.InputType==0,
      AErrorClamp=A+B*C;
   else
      AErrorClamp=A;
   end;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error-clamp trials!');
   end;
end;

if any(opts.TrialType==2),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break=opts.Abreak*A;
   B_break=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if any(opts.TrialType==3),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break_EC=opts.Abreak*AErrorClamp;
   B_break_EC=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if opts.InputType==1,
   vgain=zeros(dim_y,N);
   if any(cva_ry>0),
      error('error dependent motor noise is not compatible with InputType==1!');
   end;
else
   vgain=repmat(opts.vgain.CL,1,N);
   ec_i=any(repmat(opts.TrialType,1,2)==repmat([1 3],N,1),2);
   vgain(:,ec_i)=repmat(opts.vgain.EC,1,sum(ec_i));
end;

Dn=zeros(dim_x,dim_x,N);
An=Dn;
Bn=zeros(dim_x,dim_y,N);
Zn=zeros(dim_y,N);

% compute the expected output and its covariance matrix
y_hat=zeros(N+1,dim_y);      % the expected output
Vx=zeros(dim_x,dim_x,N+1);   % the covariance matrix of the state noise
Vy=zeros(dim_y,dim_y,N+1);   % the variance of the output
Vry_n=zeros(dim_y,dim_y,N+1);% the variance of the motor noise
Vrx_n=zeros(dim_x,dim_x,N+1);% the variance of the state noise

x_hat=zeros(N+1,dim_x);
x_hat(1,:)=x0';
y_hat(1,:)=x0'*C';
xbar=x0;

opts.TrialType(N+1)=opts.TrialType(N);  % this is only to allow type dependent Vry at t=N
if any(opts.TrialType(1)==[0,2]),
   Vry_n(:,:,1)=Vry;
else
   Vry_n(:,:,1)=VryEC;
end;
Vx(:,:,1)=Vx0;
Vy(:,:,1)=C*Vx(:,:,1)*C'+Vry_n(:,:,1);
for k=2:N+1,
   if any(opts.TrialType(k)==[0 2]),  % closed loop trial
      VryType=Vry;
   else
      VryType=VryEC;
   end;
   if opts.TrialType(k-1)==0,  % closed loop trial
      An(:,:,k-1)=A;
      Bn(:,:,k-1)=B;
      Dn(:,:,k-1)=eye(dim_x);
      Zn(:,k-1)=inp(k-1,:)';
   elseif opts.TrialType(k-1)==1,  % error clamp trial
      An(:,:,k-1)=AErrorClamp;
      Bn(:,:,k-1)=B;
      Dn(:,:,k-1)=eye(dim_x);
      Zn(:,k-1)=opts.error_clamp(k-1,:)';
   elseif opts.TrialType(k-1)==2,  % set break after closed loop trial k
      if opts.n_break(k-1)~=1
         An(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*A;
         Bn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
         Dn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1));
      else
         An(:,:,k-1)=A_break;
         Bn(:,:,k-1)=B_break;
         Dn(:,:,k-1)=opts.Abreak;
      end
      Zn(:,k-1)=inp(k-1,:)';
   elseif opts.TrialType(k-1)==3,  % set break after error clamp trial k
      if opts.n_break(k-1)~=1
         An(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*AErrorClamp;
         Bn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1))*B;
         Dn(:,:,k-1)=diag(diag(opts.Abreak).^opts.n_break(k-1));
      else
         An(:,:,k-1)=A_break_EC;
         Bn(:,:,k-1)=B_break_EC;
         Dn(:,:,k-1)=opts.Abreak;
      end
      Zn(:,k-1)=opts.error_clamp(k-1,:)';
   end;
   
   
   if any(cva_ry>0),
      
      % update the motor noise covariance matrix
      if any(opts.TrialType(k-1)==[0 2]),  % closed loop trial
         err_nm1_expected=inp(k-1,:)'-C*xbar;
         if opts.signed_EDPMN,
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C*Vx(:,:,k-1)*C';
            cv_expected_err=cv_expected_cx+diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
            for inp_i=1:size(inp,2),
               m_cx_err=[C(inp_i,:)*xbar;err_nm1_expected(inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),-cv_expected_cx(inp_i,inp_i);  -cv_expected_cx(inp_i,inp_i),cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end;
         else
            expected_err_sq=C*Vx(:,:,k-1)*C'+ err_nm1_expected*err_nm1_expected'+diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
         end;
      else
         if opts.signed_EDPMN,
            expected_err_sq=zeros(size(inp,2));
            cv_expected_cx=C*Vx(:,:,k-1)*C';
            cv_expected_err=diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
            for inp_i=1:size(inp,2),
               m_cx_err=[C(inp_i,:)*xbar;opts.error_clamp(k-1,inp_i)];
               S_cx_err=[cv_expected_cx(inp_i,inp_i),0;  0,cv_expected_err(inp_i,inp_i)];
               expected_err_sq(inp_i,inp_i)=msqy_quadrant13(m_cx_err,S_cx_err);
            end;
         else
            expected_err_sq=opts.error_clamp(k-1,:)'*opts.error_clamp(k-1,:) +diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1));
         end;
      end;
      Vry_n(:,:,k)=VryType+diag(cva_ry.^2.*diag(expected_err_sq));
   else
      Vry_n(:,:,k)=VryType;
   end;
   
   
   Vrx_n(:,:,k-1)=Vrx+diag(cva_rx.^2.*(diag(Vx(:,:,k-1))+xbar.^2));
   xbar=An(:,:,k-1)*xbar+Bn(:,:,k-1)*Zn(:,k-1);
   Vx_tmp=An(:,:,k-1)*Vx(:,:,k-1)*An(:,:,k-1)'+Dn(:,:,k-1)*Vrx_n(:,:,k-1)*Dn(:,:,k-1)' + Bn(:,:,k-1)*diag(vgain(:,k-1))*Vry_n(:,:,k-1)*diag(vgain(:,k-1))*Bn(:,:,k-1)';
   %** clip Vx: ****
   Vx_tmp(abs(Vx_tmp)>2000)=(2*(Vx_tmp(abs(Vx_tmp)>2000)>0)-1)*2000;
   Vx(:,:,k)=Vx_tmp;
   
   x_hat(k,:)=xbar';
   y_hat(k,:)=xbar'*C';
   
   Vy(:,:,k)=C*Vx(:,:,k)*C'+Vry_n(:,:,k);
end;

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

if nargin<3 || (~isempty(cva_ry) && all(cva_ry==0)),
   cva_ry=[];
end;

if isempty(cva_ry),
   M=get_noise_diffEQ_sysmatrix(A,cva_rx);
else
   dim_y=length(cva_ry);
   M=get_noise_diffEQ_sysmatrix_with_ErrorDependentMotorNoise(A,cva_rx,B,vgain,cva_ry,isErrorClamp);
end;



[V,L]=eig(M);
L=diag(L);
if nargout<2,
   return;
end;

dim_x=size(A,1);
if any(abs(L)>=1),
   % instable: reduce cva_rx until stability is achieved
   while any(cva_rx>=1e-5) || (~isempty(cva_ry) && any(cva_ry>=1e-5) ),
      for k=1:dim_x,
         if cva_rx(k)>=1e-5,
            cva_rx(k)=cva_rx(k)*0.8;
         else
            cva_rx(k)=0;
         end;
      end;
      
      if ~isempty(cva_ry) && any(cva_ry>=1e-5),
         for k=1:dim_y,
            if cva_ry(k)>=1e-5,
               cva_ry(k)=cva_ry(k)*0.8;
            else
               cva_ry(k)=0;
            end;
         end;
      end;
      if isempty(cva_ry),
         L=analyse_asymptotic_state_cov(A,cva_rx);
      else
         L=analyse_asymptotic_state_cov(A,cva_rx,cva_ry,isErrorClamp,B,vgain);
      end;
      if all(abs(L)<1),
         break;
      end;
   end;
   if any(abs(L)>=1),
      Vxx_Asymp=inf(dim_x,dim_x);
      return;
   end;
   if isempty(cva_ry),
      M=get_noise_diffEQ_sysmatrix(A,cva_rx);
   else
      M=get_noise_diffEQ_sysmatrix_with_ErrorDependentMotorNoise(A,cva_rx,B,vgain,cva_ry,isErrorClamp);
   end;
end;

vgain=vgain(:);
dim_u=length(vgain);
if dim_u~=size(B,2),
   error('invalid dimension of vgain!!');
end;

dim_v=round(dim_x*(dim_x+1)/2);

if isempty(cva_ry),
   VTrans=B*diag(vgain)*Vry*diag(vgain)*B';
else
   S2_ry_0=diag(Vry);
   VTrans=zeros(dim_x,dim_x);
end;

x_inf=(eye(dim_x)-A)\B*u_inf;

if isempty(cva_ry),
   MC=zeros(dim_v,1);
else
   MC=zeros(dim_v+dim_y,1);
end;

v1=0;
for i1=1:dim_x,
   for j1=i1:dim_x,
      v1=v1+1;
      if i1==j1,
         MC(v1)=MC(v1)+cva_rx(i1)^2*x_inf(i1)^2;
      end;
      MC(v1)=MC(v1)+Vrx(i1,j1)+VTrans(i1,j1);
   end;
end;

if ~isempty(cva_ry),
   for i1=1:dim_y,
      if isErrorClamp,
         MC(dim_v+i1)=S2_ry_0(i1)+cva_ry(i1)^2*u_inf(i1)^2;
      else
         MC(dim_v+i1)=S2_ry_0(i1)+cva_ry(i1)^2*(u_inf(i1)-sum(x_inf))^2;
      end;
   end;
end;


vs=(eye(size(M,1))-M)\MC;
%vs-M*vs-MC
v1=0;
Vxx_Asymp=zeros(dim_x,dim_x);
for i1=1:dim_x,
   for j1=i1:dim_x,
      v1=v1+1;
      Vxx_Asymp(i1,j1)=vs(v1);
      if i1~=j1,
         Vxx_Asymp(j1,i1)=vs(v1);
      end;
   end;
end;
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
if nargin<10,
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
   end;
end;
[tmp,Vrx0]=compute_asymptotic_Autocov_y(u_inf,A_inf,B,C,vgain_inf,Vrx,cva_rx,Vry_inf,0,pars.cva_ry,isErrorClamp_inf);
end

function [x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,u,y,x0,V0,var_x,var_y,opts)
%  [x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,u,y,x0,V0,var_x,var_y,opts)
% Kalman filter and Kalman smoother for the system with the following trial types:
%
% For opts.InputType==0: (driven by input inp=u)
%  closed loop trials:          x(n+1)=      A*x(n)+B*(inp(n)+vgain.CL*v)+r                                   for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=(A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r                        for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(       A*x(n)+B*(inp(n)+vgain.CL*v)+r  )            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*( (A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r  ) for 2<opts.TrialType(n)
%
% For opts.InputType==1: (driven by error inp=u-y_observed)
%  closed loop trials:          x(n+1)=A*x(n)+B*inp(n)+r                                for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=A*x(n)+B*error_clamp(k,:)'+r                     for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*inp(n)+r)            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*error_clamp(k,:)'+r) for 2<opts.TrialType(n)
%
%  For all trial types:
%                       y(n)=C*x(n) + v
%                       r ~ N(0,var_x) : process noise
%                       v ~ N(0,var_y) : measurement noise
%                       x0~ N(0,V0)    : initial value of the states
% 
% When called without arguments, kalman_observer() runs a demo program
%
% Arguments:
% A:           System matrix       [dim_x,dim_x]
% B:           Input gain matrix   [dim_x,dim_u]
% C:           Output gain matrix  [dim_y,dim_x]
% u:           Input               [N,dim_u]
%                For dim_u==1, u may also be passed as row vector
% y:           Output observations [N,dim_y]
%                For dim_y==1, y may also be passed row vector
% x0:          The initial state. Row- or column vector with length dim_x
% V0:          The initial state covariance matrix. [dim_x,dim_x]
% var_x:       Covariance matrix [dim_x,dim_x], or time variant covariance matrix [dim_x,dim_x,N) of the process noise 
% var_y:       Covariance matrix [dim_y,dim_y], or time variant covariance matrix [dim_y,dim_y,N) of the measurement noise 
% opts:        Optional structure with the following fields:
%                 isMissing:   Logical vector of length N. isMissing(n)==true indicates that no observatin
%                              is available in trial n. In that case, y(n,:) is ignored.
%                              (default: any(isnan(y),2)    )
%                 InputType:   Scalar integer specifying the input type:
%                                 0: driven by input u
%                                 1: driven by observed error u-y 
%                              (default: 0)
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                                 1<TrialType(n)<=2: closed loop trial n is followed by a set break
%                                 2<TrialType(n)   : error clamp trial n is followed by a set break
%                              (default: zeros(N,1)  )
%                 Abreak:      The matrix that specifies the state update during the set break as
%                              defined above.
%                              (default: eye(dim_x)  )
%                 vgain:       Structure with fields .CL and .EC, containing the transfer factor of the measurement
%                              measurement noise to the input of the state update for closed loop trials and error
%                              clamp trials, respectively. A value of -1 indicates that the measurement noise is transfered 
%                              completely to the state update (default: vgain.CL=0; vgain.EC=0) 
%                              vgain is irrelevant for InputType~=0
%                 error_clamp: Numerical matrix with dimension [N,dim_u] which specifies state update
%                              in the error clamp trials as defined above. For dim_y==dim_u==1, error_clamp
%                              may also be passed as row vector.
%                              (default: zeros(N,dim_u)  )
%                     n_break: break lenght (default: ones(N,1))
%                do_smoothing: If true, the algorithm performs the backward iterative kalman smoothing as
%                              described in 
%                                     Rauch, H.E., Striebel, C., & Tung, F. (1965). 
%                                     Maximum likelihood estimates of linear dynamic systems.
%                                     AIAA journal, 3 (8), 1445-1450.
%                              In that case x, V, and V_np1_n a conditioned expectation value
%                              across all smoothed x-estimates for the given system parameters.
%                              (default: true)
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
if nargin<1,
   [x,V,V_np1_n,xK,K,VK,VK_np1_n]=test_kalman_observer();
   return;
end;

x0=x0(:);
dim_y=size(C,1);
dim_u=size(B,2);

if dim_y~=dim_u,
   error('size(B,2) ~= size(C,1)!');
end;

if size(y,1)==1 && dim_y==1,
   y=y';
end;
if size(u,1)==1 && dim_u==1,
   u=u';
end;

N=size(y,1);
dim_x=size(A,1);
siz_var_y=size(var_y);
if length(siz_var_y)==2,
   var_y=reshape(repmat(var_y,1,N),siz_var_y(1),siz_var_y(2),N);
end;

siz_var_x=size(var_x);
if length(siz_var_x)==2,
   var_x=reshape(repmat(var_x,1,N),siz_var_x(1),siz_var_x(2),N);
end;

default_opts.isMissing=false(N,1);
default_opts.InputType=0;
default_opts.TrialType=zeros(N,1);
default_opts.Abreak=eye(dim_x);
default_opts.error_clamp=zeros(N,dim_u);
default_opts.n_break=ones(N,1);
default_opts.do_smoothing=true;
default_opts.vgain.CL=0;
default_opts.vgain.EC=0;

if nargin<10 || isempty(opts),
   opts=[];
end;
opts=set_default_parameters(opts,default_opts);


% ceil the trial types:
opts.TrialType=opts.TrialType(:);
opts.TrialType=ceil(opts.TrialType);
if any(opts.TrialType>3),
   error('TrialType may not exceed 3!');
end;
if any(opts.TrialType<0),
   error('TrialType must exceed -1!');
end;

if ~any(opts.InputType==[0 1]),
   error('invalid input type!');
end;
if size(opts.error_clamp,1)==1 && dim_y==1 && dim_u==1,
   opts.error_clamp=opts.error_clamp';
end;

if length(opts.TrialType)~=N,
   error('length(opts.TrialType) must equal the number of observations!');
end;

opts.isMissing=opts.isMissing(:);
if length(opts.isMissing)~=N,
   error('length(opts.isMissing) must equal the number of observations!');
end;
opts.isMissing=(opts.isMissing | any(isnan(y),2));

if opts.InputType==1,
   opts.vgain.CL=zeros(size(opts.vgain.CL));
   opts.vgain.EC=zeros(size(opts.vgain.EC));
end;

if any(opts.TrialType==1 | opts.TrialType==3),
   if opts.InputType==0,
      AErrorClamp=A+B*C;
   else
      AErrorClamp=A;
   end;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if any(opts.TrialType==2),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break=opts.Abreak*A;
   B_break=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if any(opts.TrialType==3),
   if any([dim_x,dim_x]~=size(opts.Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break_EC=opts.Abreak*AErrorClamp;
   B_break_EC=opts.Abreak*B;
   if size(opts.error_clamp,2)~=dim_u,
      error('dimension error with error_clamp-trials!');
   end;
end;

if opts.InputType==1,
   u=u-y;
   tmp=(1:N)';
   u=interp1(tmp(~isnan(u)),u(~isnan(u)),tmp,'linear','extrap');
   clear tmp
end;

K=NaN(N,dim_x*dim_y);    % kalman matrices
if opts.do_smoothing || nargout>7,
   V_ap=zeros(N,dim_x*dim_x);  % time update of the state-error covariance (before measurement)
   x_ap=zeros(N,dim_x);        % state estimates (before measurement)
end;
V=zeros(N,dim_x*dim_x);      % covariances of the state estimation-error (after measurement)
x=zeros(N,dim_x);            % state estimates (after measurement)
V_np1_n=NaN(N,dim_x*dim_x);  % covariances of the state estimation-errors cov(x(n+1)-x_true(n+1),x(n)-x_true(n))

Vxbar = V0; 
xbar=x0;


for k=1:N,  % index 1 corresponds to time0
   if opts.do_smoothing || nargout>7,  
      V_ap(k,:)=Vxbar(:)';  % save the apriori estimates needed for smoothing
      x_ap(k,:)=xbar';
   end;
   % measurement update
   ybar=C*xbar;
   
   
   
   
   % change system parameters according to trial type:
   if opts.TrialType(k)==0,      % closed loop trial
      inp=u(k,:)';
      A_=A;
      B_=B;
      var_x_=var_x(:,:,k);
      g=diag(opts.vgain.CL);
   elseif opts.TrialType(k)==1,  % error clamp trial
      inp=opts.error_clamp(k,:)';
      A_=AErrorClamp;
      B_=B;
      var_x_=var_x(:,:,k);
      g=diag(opts.vgain.EC);
   elseif opts.TrialType(k)==2, % set break after closed loop trial k
      inp=u(k,:)';
      if opts.n_break(k)~=1
         Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k));
         A_=Abreak_n*A;
         B_=Abreak_n*B;
         var_x_=Abreak_n*var_x(:,:,k)*Abreak_n';
      else
         A_=A_break;
         B_=B_break;
         var_x_=opts.Abreak*var_x(:,:,k)*opts.Abreak';
      end
      g=diag(opts.vgain.CL);
   else
      inp=opts.error_clamp(k,:)';
      if opts.n_break(k)~=1
         Abreak_n=diag(diag(opts.Abreak).^opts.n_break(k));
         A_=Abreak_n*AErrorClamp;
         B_=Abreak_n*B;
         var_x_=Abreak_n*var_x(:,:,k)*Abreak_n';
      else
         A_=A_break_EC;
         B_=B_break_EC;
         var_x_=opts.Abreak*var_x(:,:,k)*opts.Abreak';
      end
      g=diag(opts.vgain.EC);
   end;   
   
   
   if ~opts.isMissing(k),
      Ki=Vxbar*C'/(C*Vxbar*C'+var_y(:,:,k));   % Kalman update
      Vx   = Vxbar-Ki*C*Vxbar;          % A posteriori covariance
      xhat = xbar + Ki*(y(k,:)'-ybar);  % State estimate
      K(k,:)=Ki(:)'; 
      
      xhat_updated=xhat;
      Vx_updated=Vx;
      Km=B_*g*var_y(:,:,k)/(C*Vxbar*C'+var_y(:,:,k));
      cor_updated=Km*(y(k,:)'-ybar);
      P_1=Km*C*Vxbar*A_';
      P_2=Km*var_y(:,:,k)*g*B_';
  
   else  % no observations available at this sampling time
      xhat = xbar;                        % Copy a priori state estimate
      Vx   = Vxbar;                       % Copy a priori covariance factor
   end
   
   
   
   
   % --- Time update (a'priori update) of state and covariance ---
   
   if opts.InputType==0,
      xhat_updated=A_*xhat_updated+B_*inp;
      Vx_updated=A_*Vx_updated*A_' + var_x_ + B_*g*var_y(:,:,k)*g*B_';
      if opts.isMissing(k),
         cor_updated=A_*cor_updated;
         P_1=A_*P_1*A_';
         P_2=A_*P_2*A_';
      end;
      xbar=xhat_updated+cor_updated; % State update
      Vxbar = Vx_updated ...
         -P_1-P_1'-P_2;  % Covariance update
   else
      xbar=A_*xhat+B_*inp;                       % State update
      Vxbar = A_*Vx*A_' + var_x_;                               % Covariance update
   end;
   Vx_np1_p=A_*Vx;
      

   % --- Store results ---
   x(k,:)=xhat';
   V(k,:)= Vx(:)';
   V_np1_n(k,:)=Vx_np1_p(:)';
end;

if nargout>4,
   xK=x;
end;
if nargout>5,
   VK=V;
end;
if nargout>6,
   VK_np1_n=V_np1_n;
end;

if opts.do_smoothing,
   % run the kalman smoother
   for k=N-1:-1:1,
      Vx_t_t=reshape(V(k,:),dim_x,dim_x);
      Vx_tp1_t=reshape(V_ap(k+1,:),dim_x,dim_x);
      if opts.TrialType(k)<=0,      % closed loop trial
         J=Vx_t_t*A'/Vx_tp1_t;
      elseif opts.TrialType(k)<=1,  % error clamp trial
         J=Vx_t_t*AErrorClamp'/Vx_tp1_t;
      else                          % set break after trial k
         J=Vx_t_t*A_break'/Vx_tp1_t;
      end;
      x(k,:)=x(k,:)+(x(k+1,:)-x_ap(k+1,:))*J';
      Vx=Vx_t_t+J*(reshape(V(k+1,:),dim_x,dim_x)-Vx_tp1_t)*J';
      V(k,:)=Vx(:)';
      V_np1_n_k=reshape(V(k+1,:),dim_x,dim_x)*J';
      V_np1_n(k,:)=V_np1_n_k(:)';
   end
end;

end


function [x,V,V_np1_n,K,xf,Vf,Vf_np1_n]=test_kalman_observer()
%    This is a internal (non-exported) utility only called by kalman_observer 

a=[0.6;0.98]; B=[0.2;0.1];
C=[1 1];
A=diag(a)-B*C;
N=300;
t=(0:N-1)';

u=zeros(N,1);
u(10:end)=1;
u(190:end)=0;
y=zeros(N,1);
x0=[0;0];
var_x=zeros(2,2);
cva_rx=[0;0];
var_y=0;
ko_opts.InputType=0;
ko_opts.TrialType=zeros(N,1);
ko_opts.TrialType(200:250)=1;
dim_x=size(C,2);

if ko_opts.InputType==1,
   A=diag(a);
end;

% simulate the system without noise
ko_opts.isMissing=true(N,1);
ko_opts.do_smoothing=false;
if ko_opts.InputType==1,
   A_=A-repmat(B,1,length(a));
else
   A_=A;
end;
InputType_back=ko_opts.InputType;
ko_opts.InputType=0;
V0=zeros(2,2);

x_s=sim_noisy_system(A,B,C,u,x0,V0,var_x,zeros(2,1),0,0,NaN,1,ko_opts);
x_s=x_s(1:end-1,:);
ko_opts.InputType=InputType_back;
y_s=x_s*C';

figure(1);
clf
hold on
plot(t,C*x_s','-b')
plot(t,x_s(:,1),'-r')
plot(t,x_s(:,2),'-m')
% if ~isempty(which('StateSpaceModel_lib')),
%    % test that this gives the same results as lsim_stm_error_clamp (does NOT work with set-break trials)
%    stpl=StateSpaceModel_lib();
%    if ko_opts.InputType==0,
%       stm=ss(A,B,C,0,1);
%    else
%       stm=ss(A-repmat(B,1,length(a)),B,C,0,1);
%    end;
%    isErrorClamp=ko_opts.TrialType>0 & ko_opts.TrialType<=1;
%    [y_s1,x_s1]=stpl.lsim_stm_error_clamp(stm,u,t,isErrorClamp,x0,true);
%    fprintf('simulation test:\n err(y)=%10.4e  err(x)=%10.4e\n',max(abs(y_s-y_s1)),max(max(abs(x_s-x_s1))));
% end
% generate a noisy response:
y=y_s+randn(N,1)*0.1;
plot(t,y,'-c');
%
ko_opts.isMissing=false(N,1);
ko_opts.do_smoothing=true;
V0=eye(2)*0.1^2;
var_y=0.1;
var_x=eye(2)*1e-15; % we set this as small, since y results from a system without process noise
[x,V,V_np1_n,K,xf,Vf,Vf_np1_n]=kalman_observer(A,B,C,u,y,x0,V0,var_x,var_y,ko_opts);
plot(t,x*C','-k');
legend('y','xf','xs','y+r','y\_smoother');
set(get(gca,'XLabel'),'string','trial #');
set(get(gca,'YLabel'),'string','adaptive change');


figure(2);
clf
subplot(2,1,1);
hold on
plot(sum((xf-x_s).^2,2).^0.5,'-r')
plot(sum((x-x_s).^2,2).^0.5,'-b')
legend('kalman','smoother');
set(get(gca,'XLabel'),'string','trial #');
set(get(gca,'YLabel'),'string','norm(x-x\_hat))');
subplot(2,1,2);
hold on

d=zeros(N,1);
df=zeros(N,1);
for kk=1:N,
   d(kk)=trace(reshape(V(kk,:),dim_x,dim_x));
   df(kk)=trace(reshape(Vf(kk,:),dim_x,dim_x));
end

plot(t,df,'-r');
plot(t,d,'-b');
set(get(gca,'XLabel'),'string','trial #');
set(get(gca,'YLabel'),'string','trace(cov(x))');


% run a simulation to check V and V_np1_n
Nsim=1000;
x_sim=zeros(N,Nsim*dim_x);
x=zeros(N,dim_x);

ko_opts.do_smoothing=false;       % set this to false to check the kalman estimation errors
var_x=eye(2)*1e-3;


dim_y=size(C,1);
dim_u=size(B,2);
ko_opts.error_clamp=zeros(N,dim_u);
if any(ko_opts.TrialType>0 & ko_opts.TrialType<=1),
   if ko_opts.InputType==0,
      AErrorClamp=A+B*C;
   else
      AErrorClamp=A;
   end;
   if (dim_y~=dim_u || size(ko_opts.error_clamp,2)~=dim_u),
      error('dimension error with error_clamp-trials!');
   end;
end;

Abreak=eye(2);
if any(ko_opts.TrialType>1),
   if any([dim_x,dim_x]~=size(Abreak)),
      error('dimension of Abreak unequals [%d,%d]',dim_x,dim_x);
   end;
   A_break=Abreak*A;
   B_break=Abreak*B;
   if (dim_y~=dim_u || size(ko_opts.error_clamp,2)~=dim_u),
      error('dimension error with error_clamp-trials!');
   end;
end;

for si=1:Nsim,
   v=mvnrnd(0,var_y,N);
   r=mvnrnd([0 0],var_x,N);
   xbar=x0+mvnrnd([0 0],V0,1)'+r(1,:)';
   x(1,:)=xbar';
   y(1,:)=(C*xbar+v(1,:)')';
   for k=2:N,
      if ko_opts.TrialType(k-1)<=0,      % closed loop trial
         if ko_opts.InputType==0,
            xbar=A*xbar+B*u(k-1,:)'+r(k-1,:)';                         % State update
         else
            xbar=A*xbar+B*(u(k-1,:)-y(k-1,:))'+r(k-1,:)';              % State update
         end;
      elseif ko_opts.TrialType(k-1)<=1,  % error clamp trial
         xbar=AErrorClamp*xbar+B*ko_opts.error_clamp(k-1,:)'+r(k-1,:)';   % State update
      else                          % set break after trial k
         if ko_opts.InputType==0,
            xbar=A_break*xbar+B_break*u(k-1,:)'+Abreak*r(k-1,:)';
         else
            xbar=A_break*xbar+B_break*(u(k-1,:)-y(k-1))'+Abreak*r(k-1,:)';
         end;
      end;
      x(k,:)=xbar';
      y(k,:)=(C*xbar+v(k,:)')';
   end;
   [x_e,V,V_np1_n,K,xf,Vf,Vf_np1_n]=kalman_observer(A,B,C,u,y,x0,V0,var_x,var_y,ko_opts);
   x_sim(:,(si-1)*dim_x+(1:dim_x))=x_e-x;  % simulated estimation errors
end;


%** compute the simulated error covariances
V_sim=zeros(N,dim_x*dim_x);
V_np1_n_sim=NaN(N,dim_x*dim_x);
for k=1:N,
   z1=reshape(x_sim(k,:),dim_x,Nsim)';
   if k<N,
      z2=reshape(x_sim(k+1,:),dim_x,Nsim)';
      cv=cov([z2,z1]);
      cv1=cv(dim_x+(1:dim_x),dim_x+(1:dim_x));
      V_sim(k,:)=cv1(:)';
      cv1(:,:)=cv(1:dim_x,dim_x+(1:dim_x));
      V_np1_n_sim(k,:)=cv1(:);
   else
      cv=cov(z1);
      V_sim(k,:)=cv(:)';
   end;
end;

figure(3);
clf
subplot(2,1,1);
hold on
plot(t,x_sim(:,1:2:end),'-b')
if ko_opts.do_smoothing
   title('state-estimation errors (Kalman-Smoother)');
else
   title('state-estimation errors (Kalman-Filter)');
end;
subplot(2,1,2);
hold on
plot(t,x_sim(:,2:2:end),'-b')
set(get(gca,'XLabel'),'string','trial #');
set(get(gca,'YLabel'),'string','estimation error');
   

ylabel_str={'cv\_11','cv\_12','cv\_22'};
cv_indices=[1 3 4];
figure(4);
clf
for k=1:3,
   subplot(3,1,k);
   hold on
   plot(t,V(:,cv_indices(k)),'-b');
   plot(t,V_sim(:,cv_indices(k)),'-r');
   if k==1,
      legend('computed','simulated');
      if ko_opts.do_smoothing
         title('error covariance (Kalman-Smoother)');
      else
         title('error covariance (Kalman-Filter)');
      end;
   elseif k==3,
      set(get(gca,'XLabel'),'string','trial #');
   end;
   set(get(gca,'YLabel'),'string',ylabel_str{k});
end;

figure(5);
clf
for k=1:3,
   subplot(3,1,k);
   hold on
   plot(t,V_np1_n(:,cv_indices(k)),'-b');
   plot(t,V_np1_n_sim(:,cv_indices(k)),'-r');
   if k==1,
      legend('computed','simulated');
      if ko_opts.do_smoothing
         title('error covariance (t+1,t) (Kalman-Smoother)');
      else
         title('error covariance (t+1,t) (Kalman-Filter)');
      end;
   elseif k==3,
      set(get(gca,'XLabel'),'string','trial #');
   end;
   set(get(gca,'YLabel'),'string',ylabel_str{k});
end;

end

function [NLL,w_ssqerr,logDet]=incomplete_negLogLike_fun_old(A,B,C,y,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,opts)

if size(y,1)==1,
   y=y';
end;
[NTrials,NS]=size(y);
if size(inp,1)==1,
   inp=inp';
end;
NS=max([NS,size(inp,2)]);
if size(opts.TrialType,1)==1,
   opts.TrialType=opts.TrialType';
end;
NS=max([NS,size(opts.TrialType,2)]);

if ~isfield(opts,'error_clamp')
   opts.error_clamp=zeros(NTrials,1);
end;
if size(opts.error_clamp,1)==1,
   opts.error_clamp=opts.error_clamp';
end;
NS=max([NS,size(opts.error_clamp,2)]);

if ~isfield(opts,'n_break')
   opts.n_break=ones(NTrials,1);
end;
if size(opts.n_break,1)==1,
   opts.n_break=opts.n_break';
end;
NS=max([NS,size(opts.n_break,2)]);

if size(y,2)<NS,
   y=[y,repmat(y(:,end),1,NS-size(y,2))];
end;
if size(inp,2)<NS,
   inp=[inp,repmat(inp(:,end),1,NS-size(inp,2))];
end;
if size(opts.TrialType,2)<NS,
   opts.TrialType=[opts.TrialType,repmat(opts.TrialType(:,end),1,NS-size(opts.TrialType,2))];
end;
if size(opts.error_clamp,2)<NS,
   opts.error_clamp=[opts.error_clamp,repmat(opts.error_clamp(:,end),1,NS-size(opts.error_clamp,2))];
end;
if size(opts.n_break,2)<NS,
   opts.n_break=[opts.n_break,repmat(opts.n_break(:,end),1,NS-size(opts.n_break,2))];
end;

MCV_opts=opts;

NLL=0;
w_ssqerr=0;
logDet=0;
for dat_i=1:size(y,2),
   if dat_i==1 || max(abs(inp(:,dat_i)-inp(:,dat_i-1)))>0 ...
         || max(abs(opts.TrialType(:,dat_i)-opts.TrialType(:,dat_i-1)))>0,
      new_input=true;
      MCV_opts.TrialType=opts.TrialType(:,dat_i);
      MCV_opts.error_clamp=opts.error_clamp(:,dat_i);
      MCV_opts.n_break=opts.n_break(:,dat_i);
      [x_hat,y_hat,VyM]=MeanCovMatrix_DiscretLinearSystem(A,B,C,inp(:,dat_i),x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,MCV_opts);
      y_hat=y_hat(1:end-1);
      N=size(y,1);
      VyM=VyM(1:end-1,1:end-1);
      
   else
      new_input=false;
   end;
   isvalid=~isnan(y(:,dat_i));
   if new_input || any(isvalid~=last_isvalid),
      [V,L]=eig(VyM(isvalid,isvalid));
      L=diag(L);
      VyMI=V*diag(L.^-1)*V';
   end;
   w_ssqerr_i=(y(isvalid,dat_i)-y_hat(isvalid))'*VyMI*(y(isvalid,dat_i)-y_hat(isvalid));   
   NLL=NLL+ (w_ssqerr_i +  sum(log(L))+(N-sum(~isvalid))*log(2*pi)  )/2;
   w_ssqerr=w_ssqerr+w_ssqerr_i;
   logDet=logDet+sum(log(L));
   last_isvalid=isvalid;
end;
end


function [incomplete_negLogLike,w_ssqerr]=incomplete_negLogLike_fun_new(A,B,C,y,u,x0,V0,Vx,cva_rx,Vry,cva_ry,VryEC,opts)
%  incomplete_negLogLike=incomplete_negLogLike_fun_new(A,B,C,y,u,x0,V0,Vx,Vry,VryEC,opts)
%
% computes the incomplete negative log-likelihood of the observation y for the following time-discrete linar system:
% For opts.InputType==0: (driven by input u)
%  closed loop trials:          x(n+1)=      A*x(n)+B*(inp(n)+vgain.CL*v)+r                                   for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=(A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r                        for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(       A*x(n)+B*(inp(n)+vgain.CL*v)+r  )            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*( (A+B*C)*x(n)+B*(error_clamp(k,:)'+vgain.EC*v)+r  ) for 2<opts.TrialType(n)
%
% For opts.InputType==1: (driven by error u-y)
%  closed loop trials:          x(n+1)=A*x(n)+B*inp(n)+r                                for   opts.TrialType(n)<=0
%  error-clamp trials:          x(n+1)=A*x(n)+B*error_clamp(k,:)'+r                     for 0<opts.TrialType(n)<=1
%  set break after closed loop: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*inp(n)+r)            for 1<opts.TrialType(n)<=2
%  set break after error clamp: x(n+1)=Abreak^n_break(n)*(A*x(n)+B*error_clamp(k,:)'+r) for 2<opts.TrialType(n)
%
%  For all trial types:
%                       y(n)=C*x(n) + v
%                       r ~ N(0,Vx+diag(cva_rx.^2.*x(n).^2))         : process noise
%                       v(n) ~ N(0,VyType +diag(cva_ry.^2.*err(n-1).^2): measurement noise
%                              with err(n-1)=u(n-1)-C*x_sim(n-1)+vgain.CL*v(n-1) for closed loop trials at n-1, and
%                                   err(n-1)=error_clamp(n-1) + vgain.EC*v(n-1)  for error clamp trials at n-1.
%                              cva_ry~=0 is NOT ALLOWED for InputType==1
%                              VyType=Vry                            for closed-loop
%                                    =VyEC if ~isnan(VyEC(1,11)), otherwise =Vy  for error-clamp
%                       x0~ N(0,V0)      : initial value of the states
% 
%
% Arguments:
% A:           System matrix       [dim_x,dim_x]
% B:           Input gain matrix   [dim_x,dim_u]
% C:           Output gain matrix  [dim_y,dim_x]
% u:           Input               [N,dim_u]
%                For dim_u==1, u may also be passed as row vector
% y:           Output observations [N,dim_y]
%                For dim_y==1, y may also be passed row vector
% x0:          The initial state. Row- or column vector with length dim_x
% V0:          The initial state covariance matrix. [dim_x,dim_x]
% Vx:          Covariance matrix of the process noise [dim_x,dim_x]
% cva_rx:      coefficient of variation for each state [dim_x,1]
% cva_ry:      coefficient of variation of the motor noise [dim_y,1]
% Vry:         Covariance matrix of the measurement noise [dim_y,dim_y]
% VryEC:       if ~isnan(VryEC(1,1)): Covariance matrix of the measurement noise for error-clamp trials [dim_y,dim_y]
% opts:        optional structure with the following fields:
%                 isMissing:   Logical vector of length N. isMissing(n)==true indicates that no observation
%                              is available in trial n. In that case, y(n,:) is ignored.
%                              (default: any(isnan(y),2)    )
%                 InputType:   Scalar integer specifying the input type:
%                                 0: driven by input u
%                                 1: driven by observed error u-y 
%                              (default: 0)
%                 TrialType:   Numerical row or column vector of length N, specifying the type for each trial
%                              (see above).
%                                   TrialType(n)<=0: trial n is a closed loop trial
%                                 0<TrialType(n)<=1: trial n is an error clamp trial
%                                 1<TrialType(n)<=2: closed loop trial n is followed by a set break
%                                 2<TrialType(n)   : error clamp trial n is followed by a set break
%                              (default: zeros(N,1)  )
%                              (default: zeros(N,1)  )
%                 Abreak:      The matrix that specifies the state update during the set break as
%                              defined above.
%                              (default: eye(dim_x)  )
%                 vgain:       Structure with fields .CL and .EC, containing the transfer factor of the measurement
%                              measurement noise to the input of the state update for closed loop trials and error
%                              clamp trials, respectively. A value of -1 indicates that the measurement noise is transfered 
%                              completely to the state update (default: vgain.CL=0; vgain.EC=0) 
%                              vgain is irrelevant for InputType~=0
%                 error_clamp: Numerical matrix with dimension [N,dim_u] which specifies state update
%                              in the error clamp trials as defined above. For dim_y==dim_u==1, error_clamp
%                              may also be passed as row vector.
%                              (default: zeros(N,dim_u)  )
%                     n_break: break lenght (default: ones(N,1))
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
[x_n,Vx_n,Vrx_n,Vry_n]=expected_DiscreteSystem_stats(A,B,C,u,x0,V0,Vx,cva_rx,Vry,cva_ry,VryEC,opts);
[x,V,V_np1_n,K,xK,VK,VK_np1_n,x_ap,V_ap]=kalman_observer(A,B,C,u,y,x0,V0,Vrx_n,Vry_n,opts);

N=size(x,1);
%*** we have to repeat some of the default parameter setting
default_opts.isMissing=false(N,1);
if nargin<12 || isempty(opts),
   opts=[];
end;
opts=set_default_parameters(opts,default_opts);

opts.isMissing=opts.isMissing(:);
if length(opts.isMissing)~=N,
   error('length(opts.isMissing) must equal the number of observations!');
end;
opts.isMissing=(opts.isMissing | any(isnan(y),2));
%****


dim_x=size(A,1);
dim_y=size(C,1);

incomplete_negLogLike=(N-sum(opts.isMissing))*log(2*pi);
w_ssqerr=0;
for k=1:N,
   if opts.isMissing(k),
      continue;
   end;
   Vk=reshape(V_ap(k,:),dim_x,dim_x);
   SIGMA=C*Vk*C'+Vry_n(:,:,k);
   if dim_y>1,
      [EV,L]=eig(SIGMA);
      L=diag(L);
      DetSigma=prod(L);
   else
      EV=1;
      L=SIGMA;
      DetSigma=L;
   end;
   mu=C*x_ap(k,:)';
   w_ssqerr_i=(y(k,:)-mu')*(EV*diag(1./L)*EV')*(y(k,:)'-mu);
   incomplete_negLogLike=incomplete_negLogLike + w_ssqerr_i + log(DetSigma);
   w_ssqerr=w_ssqerr+w_ssqerr_i;
end;
incomplete_negLogLike=incomplete_negLogLike/2;

end





