% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
% Demo-script performing the fit of model M4 as described in
% Eggert T., Henriques D.Y.P., ‘t Hart B.M., Straube A. 
%             "Modeling inter-trial variability of pointing movements during visuomotor adaptation"

if isempty(which('set_default_parameters.m')),
   addpath([fileparts(which('DemoClosedLoopSystemIdentification.m')),filesep,'Utils']);
end;




hstr=input('Type ''1''/''2'' for fitting data from gradual/abrupt training: ','s');
hstr=strtrim(hstr);
if isempty(hstr)
   hstr='1';
end
if strcmpi('1',hstr(1))
   Data_gradual=[];
   load('DemoData.mat','Data_gradual');
   Data=Data_gradual;
else
   Data_abrupt=[];
   load('DemoData.mat','Data_abrupt');
   Data=Data_abrupt;
end;


% set the initial values for the fitting procedure
pars_start.xf_start=0;
pars_start.xs_start=0;
pars_start.af=0.78;
pars_start.as=0.99;
pars_start.bf=0.26;
pars_start.bs=0.1;
pars_start.abreak_exponent=5;

pars_start.vgain_CL=-1;
pars_start.vgain_EC=0;
pars_start.Vry=1;   
pars_start.Vrx11=0.0;   
pars_start.Vrx12=0.0;   
pars_start.Vrx22=0.0;   
pars_start.cva_xf=0;
pars_start.cva_xs=0;
pars_start.cva_ry=0;
pars_start.VryEC=NaN;



% determine which parameters to be fitted
do_fit.xf_start=false;
do_fit.xs_start=true;
do_fit.af=true;
do_fit.as=true;
do_fit.bf=true;
do_fit.bs=true;
do_fit.Vry=true;
do_fit.cva_xf=true;
do_fit.cva_xs=false;
do_fit.Vrx11=true;
do_fit.Vrx22=true;
do_fit.Vrx12=false;

pars_start.Vrx11=0;
pars_start.Vrx22=0;
pars_start.Vrx12=0;

do_fit.vgain_CL=true;
do_fit.vgain_EC=true;
Data.constraint_options.fix_vgainECMinusCL=1;  % set this to NaN to disable the constraint
Data.constraint_options.Vrx11EqVrx22=true;     % set this to NaN to disable the constraint

Data.Vrx0=[];     % if isempty(Data.Vrx0), the initial value of the state variance will be computed as the stationary solution


ssfit_options=[];
ssfit_options.no_fit=false;


[pars,incomplete_negLogLike,aic,w_ssqerr]=fit_LikelihoodDiscreteLinearSystem(Data,pars_start,do_fit,ssfit_options);



dlsl=DiscreteLinearSystem_lib();

% simulate the responses:
B=[pars.bf;pars.bs];
model_order=length(B);
C=ones(1,model_order);
A=diag([pars.af;pars.as])-B*C;
x0=[pars.xf_start;pars.xs_start];
Vrx=[pars.Vrx11,pars.Vrx12;pars.Vrx12,pars.Vrx22];
cva_rx=[pars.cva_xf;pars.cva_xs];
fitted_x_state0_=[pars.xf_start;pars.xs_start];
if ~isempty(Data.Vrx0),
   Vrx0=Data.Vrx0;
else
   isErrorClamp_inf=(any(Data.TrialType(1)==[1 3]));
   if Data.InputType==1,
      u_inf=0;
      A_inf=A+B*C;
   else
      if any(Data.TrialType(1)==[1 3]),
         u_inf=0;
         A_inf=A+B*C;
      else
         u_inf=Data.u(1);
         A_inf=A;
      end;
   end;
   [tmp,Vrx0]=dlsl.compute_asymptotic_Autocov_y(u_inf,A_inf    ,B,C,pars.vgain_CL,Vrx,cva_rx,pars.Vry,0,pars.cva_ry,isErrorClamp_inf);
end;

fopts.vgain.EC=pars.vgain_EC;
fopts.vgain.CL=pars.vgain_CL;
fopts.TrialType=Data.TrialType;
fopts.Abreak=diag([pars.af;pars.as].^pars.abreak_exponent);
[x_sim,y_sim]=dlsl.sim_noisy_system(A,B,C,Data.u,x0,zeros(model_order),zeros(model_order),zeros(model_order,1),0,0,NaN,1,fopts);
x_s1=x_sim(1:end-1,:);
y_s1=y_sim(1:end-1);

% ** compute the variance/covariance matrix of the observation:
[x_hat,y_hat,cv_expected]=dlsl.MeanCovMatrix_DiscretLinearSystem(A,B,C,Data.u,x0,Vrx0,Vrx,cva_rx,pars.Vry,pars.cva_ry,NaN,fopts);
cv_expected=cv_expected(1:end-1,1:end-1);
v_expected=diag(cv_expected);

%
FS=12;
figure(100);
set(gcf,'Position',[25   186   725   467]);
clf
hold on
set(gca,'Linewidth',1.5,'Fontsize',FS);
t=(0:length(Data.u)-1)';
ph=plot(t,Data.y,'+b','Linewidth',1.5);
ph(4)=plot(t,x_s1(:,1),'-m','Linewidth',1.5);
ph(3)=plot(t,x_s1(:,2),'-c','Linewidth',1.5);
plot(t,y_s1-v_expected.^0.5,'--r');
plot(t,y_s1+v_expected.^0.5,'--r');
ph(2)=plot(t,y_s1,'-r','Linewidth',1.5);
utmp=Data.u;
utmp(Data.TrialType==1 | Data.TrialType==3)=NaN;
ph(5)=plot(t,utmp,'-k','Linewidth',1.5);

lh=legend(ph,'data','model M4','slow','fast','visuomotor distorsion');
set(lh,'Position',[0.6988    0.7250    0.2579    0.2371]);
set(get(gca,'XLabel'),'string','trial #','Fontsize',FS);
set(get(gca,'YLabel'),'string','relative pointing direction (deg)','Fontsize',FS);
