% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
%    [pars,incomplete_negLogLike,aic,w_ssqerr]=fit_LikelihoodDiscreteLinearSystem(Data,x0,do_fit,options)
% Maximum-likelihood fit of a time-discrete linear system observed under closed loop conditions
function [pars,incomplete_negLogLike,aic,w_ssqerr]=fit_LikelihoodDiscreteLinearSystem(Data,x0,do_fit,options)
%  Data: struct with fields
%        Data.y
%        Data.u
%        Data.TrialType
%        Data.InputType
%        Data.signed_EDPMN
%        Data.Vrx0;
%        Data.constraint_options


dpars=default_pars();
opts.parnames=fieldnames(dpars);
Npars=length(opts.parnames);

default_do_fit=dpars;
if nargin<3 || isempty(do_fit),
   do_fit.Vry=true;
end;
for k=1:Npars,
   default_do_fit.(opts.parnames{k})=false;
end
do_fit=set_default_parameters(do_fit,default_do_fit);

if nargin<2 || isempty(x0),
   x0=dpars;
end;
x0=set_default_parameters(x0,dpars);
x0=default_pars(x0);

% ** preprocessing on Data: ***********************************************
if size(Data.y,1)==1,
   Data.y=Data.y';
end;
NS=size(Data.y,2);
if size(Data.u,1)==1,
   Data.u=Data.u';
end;
NS=max([NS,size(Data.u,2)]);
if size(Data.TrialType,1)==1,
   Data.TrialType=Data.TrialType';
end;
if ~isfield(Data,'InputType'),
   Data.InputType=0;
end;
NS=max([NS,size(Data.TrialType,2)]);
if size(Data.y,2)<NS,
   Data.y=[Data.y,repmat(Data.y(:,end),1,NS-size(Data.y,2))];
end;
if size(Data.u,2)<NS,
   Data.u=[Data.u,repmat(Data.u(:,end),1,NS-size(Data.u,2))];
end;
if size(Data.TrialType,2)<NS,
   Data.TrialType=[Data.TrialType,repmat(Data.TrialType(:,end),1,NS-size(Data.TrialType,2))];
end;

if ~isfield(Data,'signed_EDPMN'),
   Data.signed_EDPMN=false;
end;

default_constraint_options.Vrx11EqVrx22=NaN;
default_constraint_options.fix_vgainECMinusCL=NaN;
default_constraint_options.af_per_bf_slope=NaN;
default_constraint_options.af_per_bf_offs=NaN;
if ~isfield(Data,'constraint_options'),
   Data.constraint_options=[];
end
Data.constraint_options=set_default_parameters(Data.constraint_options,default_constraint_options);
if ~isnan(Data.constraint_options.fix_vgainECMinusCL) && ...
      ~(do_fit.vgain_EC && do_fit.vgain_CL),
   error('fix_vgainECMinusCL may set only with both do_fit.vgain_EC==true and do_fit.vgain_CL==true!');
end;

if ~isnan(Data.constraint_options.af_per_bf_slope) || ~isnan(Data.constraint_options.af_per_bf_offs),
   if isnan(Data.constraint_options.af_per_bf_slope) || isnan(Data.constraint_options.af_per_bf_offs),
      error('Either both or none of [af_per_bf_offs, af_per_bf_slope] must be NaN!!');
   end;
   if ~(do_fit.af && do_fit.bf),
      error('af_per_bf_slope may set only with both do_fit.af==true and do_fit.bf==true!');
   end;
end;

if ~isnan(Data.constraint_options.Vrx11EqVrx22),
   if ~(do_fit.Vrx11 && do_fit.Vrx22),
      error('Vrx11EqVrx22 may set only with both do_fit.Vrx11==true and do_fit.Vrx11==true!');
   end;
end;

if ~isfield(Data,'Vrx0'),   % if Vrx0==[], Vrx0 will be set to the stationary solution of the first section for each parameter set
   Data.Vrx0=[];
end

%********************************************************

default_ll.xf_start=-1e5;
default_ll.xs_start=-1e5;
default_ll.af=1e-5;
default_ll.as=1e-5;
default_ll.bf=1e-5;
default_ll.bs=1e-5;
default_ll.abreak_exponent=0.1;
default_ll.vgain_CL=-1;
default_ll.vgain_EC=-1;
default_ll.Vry=1e-5;   
default_ll.Vrx11=1e-5;   
default_ll.Vrx12=-500;   
default_ll.Vrx22=1e-5;
default_ll.cva_xf=0.0;
default_ll.cva_xs=0.0;
default_ll.cva_ry=0.0;
default_ll.VryEC=0.0;

default_ul.xf_start=1e5;
default_ul.xs_start=1e5;
default_ul.af=1-1e-5;
default_ul.as=1-1e-5;
default_ul.bf=2;
default_ul.bs=2;
default_ul.abreak_exponent=100;
default_ul.vgain_CL=1;
default_ul.vgain_EC=1;
default_ul.Vry=500;   
default_ul.Vrx11=500;   
default_ul.Vrx12=500;   
default_ul.Vrx22=500;   
default_ul.cva_xf=1e5;
default_ul.cva_xs=1e5;
default_ul.cva_ry=1e5;
default_ul.VryEC=500;


default_options.ll=default_ll;
default_options.ul=default_ul;
default_options.TolX=1e-13;
default_options.TolFun=1e-13;
default_options.no_fit=false;

if nargin<4 || isempty(options),
   options.ll=default_ll;
end;

options_names=fieldnames(options);
if ~isnan(strmatch_array('ll',options_names,false))
   options.ll=set_default_parameters(options.ll,default_ll);
end;

if ~isnan(strmatch_array('ul',options_names,false))
   options.ul=set_default_parameters(options.ul,default_ul);
end;

options=set_default_parameters(options,default_options);



opts.isfit=false(Npars,1);
opts.x0_full=zeros(Npars,1);
ul_full=zeros(Npars,1);
ll_full=zeros(Npars,1);
for k=1:Npars,
   opts.isfit(k)=do_fit.(opts.parnames{k});
   opts.x0_full(k)=x0.(opts.parnames{k});
   ul_full(k)=options.ul.(opts.parnames{k});
   ll_full(k)=options.ll.(opts.parnames{k});
end

opts.Data=Data;

x0_part=opts.x0_full(opts.isfit);
ul=ul_full(opts.isfit);
ll=ll_full(opts.isfit);

algorithm_str='interior-point';

fmincon_options = optimset(...
   'Display', 'iter'  ... 'final'  ... 'off' ...
  ,'DerivativeCheck', 'off' ... % Check gradients.
  ,'GradObj', 'off' ...   % Gradient of objective is provided.
  ,'Hessian','off' ...    % Hessian of objective is provided.
  ,'GradConstr', 'off' ...% Gradient of constraints is provided.
  ,'TolX',options.TolX ...
  ,'TolFun',options.TolFun ... % CHANGE FOR SIMULATION it was 1e-10
  ,'MaxIter',250 ...
  ,'LargeScale','on' ...
   );   
mversion=sscanf(version('-release'),'%d%c');
if mversion(1)>=2008,
   if mversion(1)<=2009,
      fmincon_options=optimset(fmincon_options,'LevenbergMarquardt','on');
   end;
   fmincon_options=optimset(fmincon_options,'Algorithm',algorithm_str);
else
   fmincon_options=optimset(fmincon_options,'LargeScale','off');
end;

A=[];
b=[];
Aeq=[];
beq=[];

w_ssqerr=[];
if options.no_fit,
   [incomplete_negLogLike,w_ssqerr]=DiscreteLinearSystem_fiterr_fun(x0_part,opts);
   pars_fitted=x0_part;
else
   [pars_fitted,incomplete_negLogLike]=fmincon(@DiscreteLinearSystem_fiterr_fun,x0_part,A,b,Aeq,beq ...
      ,ll,ul ...
      ,@DiscreteLinearSystem_fit_constr,fmincon_options,opts);
end;

pars=[];
x_fitted_full=zeros(length(opts.parnames),1);
j=0;
for k=1:length(opts.parnames),
   if opts.isfit(k),
      j=j+1;
      pars.(opts.parnames{k})=pars_fitted(j);
      x_fitted_full(k)=pars_fitted(j);
   else
      pars.(opts.parnames{k})=opts.x0_full(k);
      x_fitted_full(k)=opts.x0_full(k);
   end;
end;
pars=default_pars(pars);


if isempty(w_ssqerr),
   if any(opts.isfit([14,15,16])), % enforce stability of variance if fmincon converged at an infeasible point
      dlsl=DiscreteLinearSystem_lib();

      B=[pars.bf;pars.bs];
      A=diag([pars.af;pars.as]);
      cva_rx=[pars.cva_xf;pars.cva_xs];
      while any(cva_rx>=1e-5) || pars.cva_ry>1e-5,
         L=dlsl.analyse_asymptotic_state_cov(A,cva_rx,pars.cva_ry,true,B,pars.vgain_EC);
         if all(abs(L)<1),
            break;
         end;
         if pars.cva_xf>=1e-5,
            pars.cva_xf=pars.cva_xf*0.8;
            x_fitted_full(14)=pars.cva_xf;
         else
            pars.cva_xf=0;
            x_fitted_full(14)=pars.cva_xf;
         end;
         if pars.cva_xs>=1e-5,
            pars.cva_xs=pars.cva_xs*0.8;
            x_fitted_full(15)=pars.cva_xs;
         else
            pars.cva_xs=0;
            x_fitted_full(15)=pars.cva_xs;
         end;
         cva_rx=[pars.cva_xf;pars.cva_xs];
         
         if pars.cva_ry>=1e-5,
            pars.cva_ry=pars.cva_ry*0.8;
            x_fitted_full(16)=pars.cva_ry;
         else
            pars.cva_ry=0;
            x_fitted_full(16)=pars.cva_ry;
         end;
         
      end;
   end;

   
   [incomplete_negLogLike,w_ssqerr]=DiscreteLinearSystem_fiterr_fun(x_fitted_full(opts.isfit),opts);
end;
   
fitdim=sum(opts.isfit);
if ~isnan(Data.constraint_options.Vrx11EqVrx22),
   fitdim=fitdim-1;
end;
if ~isnan(Data.constraint_options.fix_vgainECMinusCL),
   fitdim=fitdim-1;
end;
if ~isnan(Data.constraint_options.af_per_bf_slope),
   fitdim=fitdim-1;
end;
% N_observations=2*sum(~isnan(Data.y));
aic=2*fitdim+2*incomplete_negLogLike; %N_observations*(log(2*pi)+1)+ N_observations*log(msq_err); % note that msq_err must equal ssq_err/N_observations
end

function [c,ceq] = DiscreteLinearSystem_fit_constr(x,opts)
j=0;
for k=1:length(opts.parnames),
   if opts.isfit(k),
      j=j+1;
      pars.(opts.parnames{k})=x(j);
   else
      pars.(opts.parnames{k})=opts.x0_full(k);
   end;
end;
pars=default_pars(pars);
ceq=[];

c=[];

dlsl=DiscreteLinearSystem_lib();
if any(opts.isfit(3:6)),
   a=diag([pars.af;pars.as])-repmat([pars.bf;pars.bs],1,2);
   %characteristic polynom:
   %   (a(1,1)-x)*(a(2,2)-x)-a(1,2)*a(2,1)
   %  =x^2-(a(1,1)+a(2,2))*x+a(1,1)*a(2,2)-a(1,2)*a(2,1)
   d=(a(1,1)+a(2,2))/2;
   D=d^2-det(a);
   if D>0,
      D2=d+D^0.5;
      D1=d-D^0.5;
      %    if abs((a(1,1)-D1)*(a(2,2)-D1)-a(1,2)*a(2,1))>1e-8 ...
      %       || abs((a(1,1)-D2)*(a(2,2)-D2)-a(1,2)*a(2,1))>1e-8,
      %       disp('****');
      %    end
      c=[c ...
        ;[abs(D1)-1+1e-5 ...
         ;abs(D2)-1+1e-5 ...
         ;-pars.af;-pars.as ...
         ;pars.af-pars.as ... 
         ] ...
        ];
      if true, % opt_p.force_positive_lambda,
         c=[c;-D1;-D2];
      end;
      if true, %opt_p.force_positive_gains,
         c=[c;-pars.bf;-pars.bs];
      end;
      if true, %opt_p.force_stable_error_clamp,
         c=[c;abs([pars.af;pars.as])-1+1e-5];
      end;
      
   else
      c = [-D;repmat(-1,4,1)];     % Compute nonlinear inequalities at x.
      if true, %opt_p.force_positive_lambda,
         c=[c;-1;-1];
      end;
      if true, %opt_p.force_positive_gains,
         c=[c;-1;-1];
      end;
      if true, %opt_p.force_stable_error_clamp,
         c=[c;-1;-1];
      end;
   end;

end;

if any(opts.isfit(11:13)),
   %    (Vrx11-L)*(Vrx22-L)-Vrx12^2=0
   % => L^2-(Vry11+Vrx22)*L+Vrx11*Vrx22-Vrx12^2=0
   % => (L-(Vry11+Vrx22)/2)^2 - (Vry11+Vrx22)^2/4 + Vrx11*Vrx22 - Vrx12^2 = 0
   % => L=(Vry11+Vrx22)/2 +- (  (Vry11+Vrx22)^2/4 - Vrx11*Vrx22 + Vrx12^2  )^0.5
   % => L=(Vry11+Vrx22)/2 +- (  (Vry11-Vrx22)^2/4 + Vrx12^2  )^0.5
   D=(pars.Vrx11-pars.Vrx22)^2/4 +pars.Vrx12^2;
   if D<0,
      c=[c;[-D;-1]];
   else
      m=(pars.Vrx11+pars.Vrx22)/2;
      D=D^0.5;
      c=[c;[-m-D; -m+D]];
   end;
   
   if ~isnan(opts.Data.constraint_options.Vrx11EqVrx22),
      ceq=[ceq;pars.Vrx11-pars.Vrx22];
   end;

end;

if any(opts.isfit(8:9)),
   if ~isnan(opts.Data.constraint_options.fix_vgainECMinusCL),
      ceq=[ceq;pars.vgain_EC-pars.vgain_CL-opts.Data.constraint_options.fix_vgainECMinusCL];
   end;
end;
if opts.isfit(14),
   c=[c;-pars.cva_xf];
end;

if opts.isfit(15),
   c=[c;-pars.cva_xs];
end;

if opts.isfit(16),
   c=[c;-pars.cva_ry];
end;

if any(opts.isfit([14,15,16])),
   B=[pars.bf;pars.bs];
   A=diag([pars.af;pars.as]);
   cva_rx=[pars.cva_xf;pars.cva_xs];

   L_EC=dlsl.analyse_asymptotic_state_cov(A,cva_rx,pars.cva_ry,true,B,pars.vgain_EC);
   L_CL=dlsl.analyse_asymptotic_state_cov(A-B*ones(1,size(A,1)),cva_rx,pars.cva_ry,false,B,pars.vgain_CL);
   
   L_EC=abs(L_EC);
   L_CL=abs(L_CL);
   
   c=[c;max(L_EC,L_CL)-1+1e-5];  
end;

if any(opts.isfit([3 5])),
   if ~isnan(opts.Data.constraint_options.af_per_bf_slope),
      ceq=[ceq;pars.af-opts.Data.constraint_options.af_per_bf_offs-pars.bf*opts.Data.constraint_options.af_per_bf_slope];
   end;
end;
end

function [incomplete_negLogLike,w_ssqerr]=DiscreteLinearSystem_fiterr_fun(x,opts)

j=0;
for k=1:length(opts.parnames),
   if opts.isfit(k),
      j=j+1;
      pars.(opts.parnames{k})=x(j);
   else
      pars.(opts.parnames{k})=opts.x0_full(k);
   end;
end;
pars=default_pars(pars);
Vrx=[pars.Vrx11,pars.Vrx12;pars.Vrx12,pars.Vrx22];

x0=[pars.xf_start;pars.xs_start];
cva_rx=[pars.cva_xf;pars.cva_xs];
B=[pars.bf;pars.bs];
if opts.Data.InputType==1,
   A=diag([pars.af;pars.as]);
else
   A=diag([pars.af;pars.as])-repmat(B,1,2);
end;
C=[1,1];


fopts.vgain.EC=pars.vgain_EC;
fopts.vgain.CL=pars.vgain_CL;
fopts.InputType=opts.Data.InputType;
fopts.signed_EDPMN=opts.Data.signed_EDPMN;
fopts.Abreak=diag([pars.af;pars.as].^pars.abreak_exponent);

dlsl=DiscreteLinearSystem_lib();
incomplete_negLogLike=0;
w_ssqerr=0;
for data_i=1:size(opts.Data.TrialType,2),
   if ~isempty(opts.Data.Vrx0),
      Vrx0=opts.Data.Vrx0;
   elseif data_i==1 || opts.Data.TrialType(1,data_i)~=opts.Data.TrialType(1,data_i-1) || opts.Data.u(1,data_i)~=opts.Data.u(1,data_i-1),
      Vrx0=dlsl.compute_asymptotic_start_Vrx(pars,opts.Data,data_i,A,B,C,Vrx,cva_rx);

      if isinf(Vrx0(1,1)),
         Vrx0=zeros(size(Vrx0));
      end;
   end;
   
   fopts.TrialType=opts.Data.TrialType(:,data_i);
   y_valid=opts.Data.y(:,data_i);
   u_valid=opts.Data.u(:,data_i);
   
   first_isvalid=find(~isnan(y_valid),1,'first');
   fopts.TrialType=fopts.TrialType(first_isvalid:end);
   y_valid=y_valid(first_isvalid:end);
   u_valid=u_valid(first_isvalid:end);
   
   [incomplete_negLogLike_i,w_ssqerr_i]=dlsl.incomplete_negLogLike_fun_new(A,B,C,y_valid,u_valid,x0,Vrx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,pars.VryEC,fopts);
   incomplete_negLogLike=incomplete_negLogLike+incomplete_negLogLike_i;
   w_ssqerr=w_ssqerr+w_ssqerr_i;
   %logDet=logDet+2*(incomplete_negLogLike_i - w_ssqerr_i - sum(~isnan(y_valid))/2*log(2*pi));
   
end;
end


function pars=default_pars(options)
if nargin<1,
   options=[];
end;
default_parameters.xf_start=0;
default_parameters.xs_start=0;
default_parameters.af=0.6093;
default_parameters.as=0.9738;
default_parameters.bf=0.3398;
default_parameters.bs=0.0650;
default_parameters.abreak_exponent=5;

default_parameters.vgain_CL=-0.5;
default_parameters.vgain_EC=0.5;
default_parameters.Vry=6;   
default_parameters.Vrx11=0.5;   
default_parameters.Vrx12=0.0;   
default_parameters.Vrx22=0.3;   
default_parameters.cva_xf=0;
default_parameters.cva_xs=0;
default_parameters.cva_ry=0;
default_parameters.VryEC=NaN;
pars=set_default_parameters(options,default_parameters);

end

