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
do_fit.cva_ry=false;
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


%************** Demo of multi-target adaptation: **************
% if true                                        % Set this to true to test multi-target adaptation.
%                                                %   All trials will be assigned to a single
%                                                %   stimulus class if ~isfield(Data,'StimulusClass')
%    Data.StimulusClass=ones(length(Data.y),1);
%    Data.StimulusClass(2:3:length(Data.y))=2;
%    Data.StimulusClass(3:3:length(Data.y))=3;
%    Data.TransferMatrix=[1.0,0.5,0.2 ...
%       ;0.5,1.0,0.5 ...
%       ;0.2,0.5,1.0];
% end
%***************************************************************

Data.Vrx0=[];     % if isempty(Data.Vrx0), the initial value of the state variance will be computed as the stationary solution


ssfit_options=[];
ssfit_options.no_fit=false;


[pars,incomplete_negLogLike,aic,w_ssqerr]=fit_LikelihoodDiscreteLinearSystem(Data,pars_start,do_fit,ssfit_options);



dlsl=DiscreteTimeVariantLinearSystem_lib();





% simulate the responses:

[A,B,C,Gv,Gr,inp,x0,Vrx,cva_rx,Vry,cva_ry,VryEC]=dlsl.mk_TimeVariantSystem(Data,pars);
if ~isempty(Data.Vrx0)
   Vrx0=Data.Vrx0;
else
   Vx0=dlsl.compute_asymptotic_start_Vrx(pars,Data,1,A(:,:,1),B(:,:,1),C(:,:,1),Vrx,cva_rx);
end
if isfield(Data,'x0')
   x0=Data.x0;
end

fopts=Data;
fopts.TrialType(fopts.TrialType==2)=0;
fopts.TrialType(fopts.TrialType==3)=1;
[x_s,y_s]=dlsl.sim_noisy_system(A,B,C,Gv,Gr,inp,x0,zeros(size(A,1)),zeros(size(A,1)),zeros(size(A,1),1),0,0,NaN,1,fopts);


% ** compute the variance/covariance matrix of the observation:
[x_hat,y_hat,cv_expected]=dlsl.MeanCovMatrix_DiscretLinearSystem(A,B,C,Gv,Gr,inp,x0,Vx0,Vrx,cva_rx,Vry,cva_ry,VryEC,fopts);
v_expected=diag(cv_expected);
         

%% ** plot the results: **********************************************************************

FS=12;
figure(101);
clf
set(gcf,'Position',[25   186   725   467]);




symb='+os<';   % leftward, rightward
cols0=[0,0,1 ...  % symbol
   ;1,0,1 ...  % fast
   ;0,1,1 ...  % slow
   ;1,0,0 ...  % model
   ];

is_StimulusClassFit=isfield(Data,'StimulusClass');
if is_StimulusClassFit
   AllSTClasses=unique(Data.StimulusClass);
   NTSClasses=length(AllSTClasses);
else
   NTSClasses=1;
end




for stim_class=1:NTSClasses
   subplot(NTSClasses,1,stim_class);
   hold on
   set(gca,'Linewidth',1.5,'Fontsize',FS);
   
   if stim_class==1
      cols=cols0;
   else
      cols=0.7*cols;
   end
   
   trial_index=(1:length(Data.y))';    % trial numbers corresponding to the elements of Data.y
                                       % In case that different models are fitted to
                                       % distinct subsets of trials, trial_index by be
                                       % discontinuous (i.e., ~all(diff(trial_index)==1)  )
   if ~is_StimulusClassFit
      ind_STClassTrial=(1:size(y_s,1))';% index of the respective stimulus class in Data.y
   else
      ind_STClassTrial=find(Data.StimulusClass==AllSTClasses(stim_class));
      trial_index=trial_index(ind_STClassTrial);
   end
   x_s_class=x_s(ind_STClassTrial,:);
   y_s_class=y_s(ind_STClassTrial);
   v_expected_class=v_expected(ind_STClassTrial);
   
   t=(trial_index(1):trial_index(end))'-1;
   
   plt_tmp=NaN(length(t),1);
   plt_tmp(trial_index-trial_index(1)+1)=Data.y(ind_STClassTrial);
   ph=plot(t,plt_tmp,[symb(stim_class),'b'],'Linewidth',1.5);
   set(ph,'MarkerEdgeColor',cols(1,:));
   
   if is_StimulusClassFit
      if pars.bf==0 && pars.Vrx11==0 && pars.Vrx12==0
         plt_tmp=zeros(length(t),1);
      else
         plt_tmp=interp1(trial_index-1,x_s_class(:,(stim_class-1)*2+1),t,'linear','extrap');
      end
   else
      plt_tmp=interp1(trial_index-1,x_s_class(:,1),t,'linear','extrap');
   end
   ph(4)=plot(t,plt_tmp,'-m','Linewidth',1.5);
   set(ph(4),'Color',cols(2,:));
   
   if is_StimulusClassFit
      if pars.bf==0 && pars.Vrx11==0 && pars.Vrx12==0
         plt_tmp=interp1(trial_index-1,x_s_class(:,stim_class),t,'linear','extrap');
      else
         plt_tmp=interp1(trial_index-1,x_s_class(:,(stim_class-1)*2+2),t,'linear','extrap');
      end
   else
      plt_tmp=interp1(trial_index-1,x_s_class(:,2),t,'linear','extrap');
   end
   ph(3)=plot(t,plt_tmp,'-c','Linewidth',1.5);
   set(ph(3),'Color',cols(3,:));
   
   plt_tmp=interp1(trial_index-1,y_s_class-v_expected_class.^0.5,t,'linear','extrap');
   ph_tmp=plot(t,plt_tmp,'--r');
   set(ph_tmp,'Color',cols(4,:));
   
   plt_tmp=interp1(trial_index-1,y_s_class+v_expected_class.^0.5,t,'linear','extrap');
   ph_tmp=plot(t,plt_tmp,'--r');
   set(ph_tmp,'Color',cols(4,:));
   
   plt_tmp=interp1(trial_index-1,y_s_class,t,'linear','extrap');
   ph(2)=plot(t,plt_tmp,'-r','Linewidth',1.5);
   set(ph(2),'Color',cols(4,:));
   
   plt_tmp=Data.u(ind_STClassTrial);
   plt_isnan=double(Data.TrialType(ind_STClassTrial)==1 | Data.TrialType(ind_STClassTrial)==3);
   plt_tmp=interp1(trial_index-1,plt_tmp,t,'linear','extrap');
   plt_isnan=(interp1(trial_index-1,plt_isnan,t,'linear','extrap')>1e-6);
   plt_tmp(plt_isnan)=NaN;
   ph(5)=plot(t,plt_tmp,'-k','Linewidth',1.5);
   
   if stim_class==NTSClasses
      lh=legend(ph,'data','model M4','slow','fast','visuomotor distorsion');
      set(lh,'Position',[0.6988    0.7250    0.2579    0.2371]);
      set(get(gca,'XLabel'),'string','trial #','Fontsize',FS);
   end
   set(get(gca,'YLabel'),'string','relative pointing direction (deg)','Fontsize',FS);
   
end





