%% JT-60SA closed_loop_simulator
clear all
close all
clc

distFlag = 1;
stabFlag = 1;
startTime = 0;
stopTime = 10; 
switchTime = startTime-0.01;
cdTime = stopTime; % no current drive
neg_time=2;
transitionTime = 1.5;
interpTime = [startTime - neg_time; startTime; startTime+transitionTime; stopTime];
%interpTimeW = [startTime - neg_time; startTime; startTime+transitionTime-1; startTime+transitionTime; stopTime];

nPF = 10;

% Number of gaps
nOfGaps = 85;
% Number of strike point
nOfStrikes = 8;
% Number of flux sensors
nOfFluxSens = 34;
% Number of magnetic sensors
nOfMagneticSens = 45;
% Number of controlled fluxes
nOfFluxCntrl = 20;
% nOfFluxCntrl = 9;
 %nOfFluxCntrl = 7;
 % :D

% Load the initial equilibrium (which is also the linear model that will
% be used for the simulation

 simConf = '../../L-models_90deg/SOF@18d66s_japanese_cntrlFluxPntsSqzd_CL.mat';



load(simConf);

% Load additional information for the script
 load('../../L-models_90deg/SOF@18d66s_japanese_cntrlFluxPntsSqzd.mat','Input_struct','x_np','y_np','y_type')


% Load the target equilibrium (if it is equal to the initial one, then no
% transitions will be performed

 targetConf = '../../L-models_90deg/SOF@18d66s_japanese_cntrlFluxPntsSqzd_CL.mat';



% targetConf = '../L-models_90deg/SOF@18d66s_1_CL.mat';

targetConf = load(targetConf);
 targetEquil = load('../../L-models_90deg/SOF@18d66s_japanese_cntrlFluxPntsSqzd.mat');


LinearModel.R(LinearModel.PlasmaCurrentInfo.StatePosition, LinearModel.PlasmaCurrentInfo.StatePosition) = 0; % force plasma resistance to 0

%% Initialize gaps
gapNames = {};
strikeNames = {};
FluxSensNames= {};
MagnSensNames={};

% Gap and strike names
for i = 1:nOfGaps
    gapNames{i} = sprintf('GAP%02d',i);
end

for i = 1:nOfStrikes
    strikeNames{i} = sprintf('GAP%02d',nOfGaps+i);
end

for i = 1:nOfFluxSens
    FluxSensNames{i} = sprintf('Flux_%03d',i);
end

for i = 1:nOfMagneticSens
    MagnSensNames{i} = sprintf('Bpol_%03d',i);
end

% for i = 1:nOfFluxCntrl
%     FluxCntrlNames{i} = sprintf('FluxCntrl_%03d',i+34);
% end

%For some reason the equilibriu, fopr 8 and 6 gaps has FluxCntrl_001
for i = 1:nOfFluxCntrl
    FluxCntrlNames{i} = sprintf('FluxCntrl_%03d',i);
end






% Gap and strike and sensors indexes
gapIdx = signalIndexByName(gapNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);
strikeIdx = signalIndexByName(strikeNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);
FluxSensIdx = signalIndexByName(FluxSensNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);
MagnSensIdx = signalIndexByName(MagnSensNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);
FluxCntrlIdx = signalIndexByName(FluxCntrlNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);


% Define gapIndex and strikeIndex, that is I'm removing all the NaN and
% negative gaps and NaN strikes
gapIndex       = [];
strikeIndex    = [];
FluxSensIndex  = [];
MagnSensIndex  = [];
FluxCntrlIndex = [];

XPntIndex      = [];

for i = 1:length(gapIdx)
    if ~isnan(LinearModel.YEquil(gapIdx(i))) && LinearModel.YEquil(gapIdx(i)) > 0
        gapIndex(end+1) = i;
    end
end

for i = 1:length(strikeIdx)
    if ~isnan(LinearModel.YEquil(strikeIdx(i))) 
        strikeIndex(end+1) = i;
    end
end

for i = 1:length(FluxSensIdx)
    if ~isnan(LinearModel.YEquil(FluxSensIdx(i))) 
       FluxSensIndex(end+1) = i;
    end
end

for i = 1:length(MagnSensIdx)
    if ~isnan(LinearModel.YEquil(MagnSensIdx(i))) 
      MagnSensIndex(end+1) = i;
    end
end

for i = 1:length(FluxCntrlIdx)
    if ~isnan(LinearModel.YEquil(FluxCntrlIdx(i))) 
      FluxCntrlIndex(end+1) = i;
    end
end



% Equilibrium values for the gaps
equilGaps      = LinearModel.YEquil(gapIdx);
% Equilibrium values for the strike 
equilStrikes   = LinearModel.YEquil(strikeIdx);
% Equilibrium values for the strike 
equilFluxSens  = LinearModel.YEquil(FluxSensIdx);
equilMagnSens  = LinearModel.YEquil(MagnSensIdx);
equilFluxCntrl = LinearModel.YEquil(FluxCntrlIdx);

targetGaps = targetConf.LinearModel.YEquil(gapIdx);
for i = 1 : nOfGaps
    Rt(i, 1) = Input_struct.r_sens_gap(i) + targetGaps(i)*cosd(Input_struct.theta_sens_gap_deg(i));
    Zt(i, 1) = Input_struct.z_sens_gap(i) + targetGaps(i)*sind(Input_struct.theta_sens_gap_deg(i));
end

%% Initialize X-point
xIdx = signalIndexByName({'CV-RX', 'CV-ZX', 'CV-FLUX'}, LinearModel.OutputsInfo.Name, 1:length(LinearModel.OutputsInfo.Name));
equilRxZx = LinearModel.YEquil(xIdx);


% Plots the first wall and the equilbrium 
close all
figure('Position', [0, 0, 500, 750])
pdemesh(Input_struct.p, Input_struct.e, []), hold on, axis([0, 6, -4.5, 4.5]), axis equal
pdecont(Input_struct.p,Input_struct.t,x_np(1:length(Input_struct.p)),y_np(strcmp(y_type,'psb_c'))*[1 1]);
xlabel('R[m]', 'Interpreter', 'latex')
ylabel('Z[m]', 'Interpreter', 'latex')
hold on

% target shape
H = pdecont(targetEquil.Input_struct.p,targetEquil.Input_struct.t,targetEquil.x_np(1:length(targetEquil.Input_struct.p))...
    ,targetEquil.y_np(strcmp(targetEquil.y_type,'psb_c'))*[1 1]);
set(H, 'Color', 'magenta')
set(H,'LineWidth',2)
set(H,'DisplayName','Limiter Plasma 3.5 [MA]')
hold on


%% Build state space model
VDNames = {'CS1','CS2','CS3','CS4','EF1','EF2','EF3','EF4','EF5','EF6','VSU','VSL','Vpl'}; 
PFNames = {'CS1','CS2','CS3','CS4','EF1','EF2','EF3','EF4','EF5','EF6','VSU','VSL'};
OutputNames = {PFNames{1:end},'Ipl',gapNames{1:end},strikeNames{1:end},'CV-RX','CV-ZX','CV-FLUX','CV-ZC',FluxSensNames{1:end},MagnSensNames{1:end},FluxCntrlNames{1:end}};
outputIdx = signalIndexByName(OutputNames,LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);

A = -inv(LinearModel.L)*LinearModel.R;
B = inv(LinearModel.L);
E = -inv(LinearModel.L)*LinearModel.LE;
% [VV,DD] = eig(A);
% ii = find(diag(DD)>0);
% DD(ii,ii) = -DD(ii,ii);
% Astab = VV*DD*inv(VV);
% 
% A_contr = Astab;
A_contr = A;
B_contr = [B(:,[1:12 end]) A_contr*E];
C_contr = LinearModel.C(outputIdx,:);
D_contr = [LinearModel.D(outputIdx,[1:12 end]) LinearModel.F(outputIdx,:)+C_contr*E];


%% Initial state
xi0 = zeros(1, size(A_contr, 1)); % substitute a VDE-related x0 if needed

%% Initialize plasma current
Ip0 = LinearModel.XEquil(LinearModel.PlasmaCurrentInfo.StatePosition);
vLoop = LinearModel.R(end,end)*Ip0;
IpIdx = signalIndexByName({'Ipl'}, OutputNames, 1:length(OutputNames));

%% Equilibrium inputs and outputs 
equilOutputs = LinearModel.YEquil(outputIdx); % from model outputs  
equilDist = LinearModel.YEquil(LinearModel.DisturbancesInfo.OutputPosition);
equilVolts = zeros(13,1);

%% Build reference waveforms for Ip and shape and scenario currents
% Ip_ref
initialIp = Ip0;
finalIp = targetConf.LinearModel.XEquil(LinearModel.PlasmaCurrentInfo.StatePosition);

Ip_ref = [startTime-neg_time initialIp;  startTime initialIp;startTime+transitionTime finalIp];
% Ip_ref = [startTime-neg_time initialIp;  startTime initialIp;startTime+transitionTime finalIp; 5 5e6];

% Gap_ref
ind1=14;
ind2=14+84;
tempOutputs = targetConf.LinearModel.YEquil(outputIdx);


all_initialGaps = equilOutputs([ind1:ind2]);
all_finalGaps = tempOutputs([ind1:ind2]);

% 
% % %% Selection of all gapso
% initialGaps = equilOutputs([ind1:ind2]);
% finalGaps = tempOutputs([ind1:ind2]);
% cntrl_gaps=(ind1:ind2);

% % 
% %%Selection of 30 gaps
% initialGaps = equilOutputs([ind1:3:(ind2-2),ind2-1,ind2]);
% finalGaps = tempOutputs([ind1:3:(ind2-2),ind2-1,ind2]);
% 
% 
% % % 
% %%Selection of 19 gaps
% initialGaps = equilOutputs([ind1:5:(ind2-2),ind2-1,ind2]);
% finalGaps = tempOutputs([ind1:5:(ind2-2),ind2-1,ind2]);
% cntrl_gaps=(ind1:5:(ind2-2))-13;

%%Selection of 20 gaps
initialGaps = equilOutputs([round((ind1+1:4.6:(ind2-2))),ind2-1,ind2]);
finalGaps = tempOutputs([round((ind1+1:4.6:(ind2-2))),ind2-1,ind2]);
cntrl_gaps=round((ind1+1:4.6:(ind2-2)))-13;
NoControlledGaps='Controlled Gaps = 20'

%%%%% Obtain R,Z coordinates of the equilibrium gaps
ii=length(Input_struct.r_sens_gap);
r_gap=Input_struct.r_sens_gap([round((1+1:4.6:(ii-10))),ii-9,ii-8]);
z_gap=Input_struct.z_sens_gap([round((1+1:4.6:(ii-10))),ii-9,ii-8]);
theta_cntrl_gap=Input_struct.theta_sens_gap_deg([round((1+1:4.6:(ii-10))),ii-9,ii-8]);
%%
r_cntrl_gap_eq=r_gap+initialGaps.*cosd(theta_cntrl_gap);
z_cntrl_gap_eq=z_gap+initialGaps.*sind(theta_cntrl_gap);
rz_cntrl_gap_sqzd=[    2.9896   -1.7907    
    3.4821   -1.3869    
    3.7765   -1.0657    
    4.0493   -0.6100    
    4.1501   -0.2475    
    4.1377    0.2836    
    3.9953    0.7430    
    3.7887    1.1142    
    3.0416    1.5178    
    2.7245    1.6830     
    2.2736    1.7400    
    1.9043    1.3408    
    1.8303    0.9160    
    1.7790    0.3850    
    1.7731   -0.0398    
    1.8149   -0.5708    
    1.9093   -1.1018    
    2.0259   -1.5266    
    2.0249   -2.3655    
    2.5353   -2.5691   ];
    %%%% Lets put sensors in the points we want to squezee the plasma
r_cntrl_gap_eq_chng=[(startTime-3) r_cntrl_gap_eq'
                       (startTime+0.99) r_cntrl_gap_eq'
                    (startTime+1) rz_cntrl_gap_sqzd(:,1)'];
 z_cntrl_gap_eq_chng=[(startTime-3) z_cntrl_gap_eq'
                      (startTime+0.9999) z_cntrl_gap_eq'
                     (startTime+1) rz_cntrl_gap_sqzd(:,2)'];
%%
% % %% Japanese selection 1 (Add one more gap at the begining)
% japs_gaps=[13,13,22,36,52,66,76,85,84]';
% japs_gaps=japs_gaps+ind1-1;
% initialGaps = equilOutputs(japs_gaps);
% finalGaps = tempOutputs(japs_gaps);
% cntrl_gaps=[13,13,22,36,52,66,76,84,85]';
% NoControlledGaps='Controlled Gaps = 8'
% %%%%% Obtain R,Z coordinates of the equilibrium gaps
% ii=length(Input_struct.r_sens_gap);
% r_gap=Input_struct.r_sens_gap([13,13,22,36,52,66,76,84,85]);
% z_gap=Input_struct.z_sens_gap([13,13,22,36,52,66,76,84,85]);
% theta_cntrl_gap=Input_struct.theta_sens_gap_deg([13,13,22,36,52,66,76,84,85]);
% 
% r_cntrl_gap_eq=r_gap+initialGaps.*cosd(theta_cntrl_gap);
% z_cntrl_gap_eq=z_gap+initialGaps.*sind(theta_cntrl_gap);
% 
% 
% % % %Japanese selection 2
% japs_gaps=[21,21,36,47,66,84,85]';
% japs_gaps=japs_gaps+ind1-1;
% initialGaps = equilOutputs(japs_gaps);
% finalGaps = tempOutputs(japs_gaps);
% cntrl_gaps=[21,21,36,47,66,84,85]';
% NoControlledGaps='Controlled Gaps = 6'
% %%%%% Obtain R,Z coordinates of the equilibrium gaps
% ii=length(Input_struct.r_sens_gap);
% r_gap=Input_struct.r_sens_gap([21,21,36,47,66,84,85]);
% z_gap=Input_struct.z_sens_gap([21,21,36,47,66,84,85]);
% theta_cntrl_gap=Input_struct.theta_sens_gap_deg([21,21,36,47,66,84,85]);
% 
% r_cntrl_gap_eq=r_gap+initialGaps.*cosd(theta_cntrl_gap);
% z_cntrl_gap_eq=z_gap+initialGaps.*sind(theta_cntrl_gap);
% 


Gap_ref = [startTime-neg_time initialGaps'; startTime initialGaps';startTime+transitionTime finalGaps'];


% I_scenario
initialPFCurr = equilOutputs([1:10]);
finalPFCurr = tempOutputs([1:10]);
IPF_scenario = [startTime-neg_time initialPFCurr';startTime initialPFCurr';startTime+transitionTime finalPFCurr'];


% Define the disturbance waveforms

%% Disruptions
load('../../Data from the Japanese/Temporal_evolutions_of_betap_and_li/li_bp_IpRU2.mat');
load('../../JT-60SA simulation scheme/disturb_PID');
%betap = [0 0*Ip0+equilDist(1); 0.1 0*Ip0+equilDist(1); 0.1+1e-3 -0.14*Ip0+equilDist(1); 0.2 -0.14*Ip0+equilDist(1)];
%li = [0 0*Ip0+equilDist(2); 0.1 0*Ip0+equilDist(2); 0.1+3e-3 -0.15*Ip0+equilDist(2); 0.2 -0.15*Ip0+equilDist(2)];

betap_ur=betap_Ur2+equilDist(1)/Ip0;
li_ur=li_Ur2+equilDist(2)/Ip0;

betap_miy=betap_Miy2+equilDist(1)/Ip0;
li_miy=li_Miy2+equilDist(2)/Ip0;

betap_ur=Ip0*betap_ur;
li_ur=Ip0*li_ur;

 li_ur=[Ura_t2;li_ur]';
 betap_ur=[Ura_t2;betap_ur]';
 
 
betap_miy=Ip0*betap_miy;
li_miy=Ip0*li_miy;
li_miy=[Miy_t2;li_miy]';
betap_miy=[Miy_t2;betap_miy]';
% 
% betap=betap_miy;
% li=li_miy;
% type_dis='Miyata'

betap=betap_ur;
li=li_ur;
type_dis='Urano'

betap = [startTime-3 betap(1,2);betap];
li = [startTime-3 li(1,2);li];
% 
% % % % %%%%%%%%%% PID page 34
%  compELM.betap(:,2)=Ip0*compELM.betap(:,2);%
%  compELM.li(:,2)=Ip0*compELM.li(:,2);
%  betap = [startTime-3 equilDist(1);startTime equilDist(1);compELM.betap];
%  li = [startTime-3 equilDist(2);startTime equilDist(2);compELM.li];
% type_dis='comp_ELM'

%  clear betap li
%  ELM.betap(:,2)=Ip0*ELM.betap(:,2);%
%  betap = [startTime-3 equilDist(1);startTime equilDist(1);ELM.betap];
%  li = [startTime-3 equilDist(2);startTime equilDist(2)];
% type_dis='ELM'
% % % %  
%  clear betap li
%  minor1.betap(:,2)=Ip0*minor1.betap(:,2);
%  minor1.li(:,2)=Ip0*minor1.li(:,2);
%  betap = [startTime-3 equilDist(1);startTime equilDist(1);minor1.betap];
%  li = [startTime-3 equilDist(2);startTime equilDist(2);minor1.li];
%  type_dis='minor_disr'
% % %  
% %   clear betap li
%  minor2.betap(:,2)=Ip0*minor2.betap(:,2);
%  
%  betap = [startTime-3 equilDist(1);startTime equilDist(1);minor2.betap];
%  li = [startTime-3 equilDist(2);startTime equilDist(2)];
 
%% Controllers 

% Load the PF current controller
load('../../JT-60SA simulation scheme/KPFcurr.mat','KPFcurr');

%% Model of the PF power supplies
% Set limits for the simulation scheme
% VSsat = 1e3;
CSsat = [1 1.3 1.3 1] * 1e3;
EFsat = [1 .97 .97 .97 .97 1] * 1e3;

I_sat_upper = [2 2 2 2 1 1 2 2 1 1]*1e4;
I_sat_lower = [-2 -2 -2 -2 -2 -2 -2 -2 -2 -2]*1e4; 

%% PFC control
PSdelay = 1.5e-3; 
PStau = 3e-3;
PSmodel = tf([1],[PStau 1]) * eye(nPF);

%% Ip control
% PF combination that produces the transformer field
kIp = -[7.7175
        6.6999
        6.8024
        7.6938
        0.3730
        0.4150
        4.5207
        2.9209
        0.7198
        0.2332]*1e3;  
kIp = kIp/norm(kIp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PID controller
IpControl.Kp = 0.001;
IpControl.Ki = 100 * IpControl.Kp;
IpControl.Kd = 0;
IpControl.Td = 1;

Ip_PID = buildPID(IpControl);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% XSC matrix
% load('../../JT-60SA simulation scheme/XSC_SOF.mat','XSC_matrix')
% load('../../XSC_SOF.mat','XSC_matrix')
load('XSC_SOF_flux.mat','XSC_matrix')

% XSC PID controller
XSC_PID.Kp =0.05* ones(1, nPF);  
XSC_PID.Ki = 200*XSC_PID.Kp;
XSC_PID.Kd = zeros(1, nPF);
XSC_PID.Td = ones(1, nPF); % dummy value

% % % XSC PID controller
% XSC_PID.Kp = 11 * ones(1, nPF);  
% XSC_PID.Ki = 450 * ones(1, nPF);
% XSC_PID.Kd = 29*ones(1, nPF);
% XSC_PID.Td = (1/2000)*ones(1, nPF); % dummy value


XSC_PI = buildPID(XSC_PID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VS system
VSL_Vsat = [-1, +1] * 1e3; % ************ In-vessel coils
VSU_Vsat = [-1, +1] * 1e3;  
VSL_Isat = [-5, +5] * 1e3; 
VSU_Isat = [-5, +5] * 1e3; 

VS.ICGain = -0.0076; % current gain
VS.ZdotGain = 20/5.5e6*Ip0; % zdot gain
VS.OverallGain = -40;
VS.Filter = tf([1, 0], [1/1000, 1]);


return

% Disturbances (not used for now)
betap.time = 0;
betap.value = 0;
li.time = 0;
li.value = 0;

initialBeta = LinearModel.YEquil(LinearModel.DisturbancesInfo.OutputPosition(1))/Ip0;
targetBeta = targetConf.LinearModel.YEquil(LinearModel.DisturbancesInfo.OutputPosition(1))/Ip0;
initialLi = LinearModel.YEquil(LinearModel.DisturbancesInfo.OutputPosition(2))/Ip0;
targetLi = targetConf.LinearModel.YEquil(LinearModel.DisturbancesInfo.OutputPosition(2))/Ip0;

%% Saturation limits
VSL_Vsat = [-1, +1] * 1e3; % ************ In-vessel coils
VSU_Vsat = [-1, +1] * 1e3;  
VSL_Isat = [-5, +5] * 1e3; 
VSU_Isat = [-5, +5] * 1e3; 

CS1_Vsat = [-1.0, +1.0] * 1e3; % ******** Central Solenoid  
CS1_Isat = [-20, +20] * 1e3;
CS2_Vsat = [-1.3, +1.3] * 1e3;   
CS2_Isat = [-20, +20] * 1e3;
CS3_Vsat = [-1.3, +1.3] * 1e3;   
CS3_Isat = [-20, +20] * 1e3;
CS4_Vsat = [-1.0, +1.0] * 1e3;   
CS4_Isat = [-20, +20] * 1e3;

EF1_Vsat = [-1.0, +1.0] * 1e3; % ******** PF coils  
EF1_Isat = [-20, +10] * 1e3; % or reverse
EF2_Vsat = [-0.97, +0.97] * 1e3;   
EF2_Isat = [-20, +10] * 1e3; % or reverse
EF3_Vsat = [-0.97, +0.97] * 1e3;   
EF3_Isat = [-20, +20] * 1e3;
EF4_Vsat = [-0.97, +0.97] * 1e3;   
EF4_Isat = [-20, +20] * 1e3;
EF5_Vsat = [-0.97, +0.97] * 1e3;   
EF5_Isat = [-20, +10] * 1e3; % or reverse
EF6_Vsat = [-1.0, +1.0] * 1e3;   
EF6_Isat = [-20, +10] * 1e3; % or reverse 

for i = 1 : length(LinearModel.PoloidalCircuits.Name)
    eval(['IsatUpperLimit(i) = ' LinearModel.PoloidalCircuits.Name{i} '_Isat(2);']);
    eval(['IsatLowerLimit(i) = ' LinearModel.PoloidalCircuits.Name{i} '_Isat(1);']);   
end

% Set limits for the simulation scheme
VSsat = 1e3;
CSsat = [1 1.3 1.3 1] * 1e3;
EFsat = [1 .97 .97 .97 .97 1] * 1e3;

%% Vertical stabilization
VS.ICGain = -0.0076; % current gain
VS.ZdotGain = 20/5.5e6*Ip0; % zdot gain
VS.OverallGain = -40;
VS.Filter = tf([1, 0], [1/1000, 1]);

%% PFC control
PSdelay = 1.5e-3; 
PStau = 3e-3;
PSmodel = tf([1],[PStau 1]) * eye(nPF);

% PID control
PFControl.Kp = 50 * ones(1, nPF);
PFControl.Ki = 10 * ones(size(PFControl.Kp));
PFControl.Kd = 0 * ones(size(PFControl.Kp));
PFControl.Td = 1e-4 * ones(size(PFControl.Kp));
PFC_PID = buildPID(PFControl);

% SFC control (Kpcurr designed on the basis of a plasmaless model)
load('Kpcurr.mat');

% Feedforward (constant)
targetCurrs = targetConf.LinearModel.XEquil(targetConf.LinearModel.PoloidalCircuits.StatePosition(1:end-2));
initialCurrs = LinearModel.XEquil(LinearModel.PoloidalCircuits.StatePosition(1:end-2));

% % Test reference
% 
% for i = 1 : nPF
%     PFCref.value(:, i) = sin(2*pi*.1*PFCref.time + i) * 100 * i;
% end

%% Shape    

% Modify C-matrix to calculate the flux at the target points
segIdx = signalIndexByName(segNames, OutputNames, 1:length(OutputNames));
C_seg_temp = C_contr(segIdx, :)/2/pi;
D_seg_temp = D_contr(segIdx, :)/2/pi;
equilSegsTemp = 1/2/pi * LinearModel.YEquil(signalIndexByName(segNames, LinearModel.OutputsInfo.Name, LinearModel.OutputsInfo.OutputPosition));

for i = 1 : nSeg
    C_temp = C_seg_temp(nPoints*(i-1) + 1 : i*nPoints, :); % Rows of C and D corresponding to the considered control segment
    D_temp = D_seg_temp(nPoints*(i-1) + 1 : i*nPoints, :);
    equilTemp = equilSegsTemp(nPoints*(i-1) + 1 : i*nPoints);

    if Ri(i) ~= Rf(i)
        r = linspace(Ri(i), Rf(i), nPoints);
        pos = interp1(r, 1:nPoints, Rt(i));
        C_seg(i, :) = interp1(r, C_temp ,Rt(i));
        D_seg(i, :) = interp1(r, D_temp ,Rt(i));
        equilSegs(i) = interp1(r, equilTemp, Rt(i));
    else % vertical segment
        z = linspace(Zi(i), Zf(i), nPoints);
        pos = interp1(z, 1:nPoints, Zt(i));
        C_seg(i, :) = interp1(z, C_temp ,Zt(i));
        D_seg(i, :) = interp1(z, D_temp ,Zt(i));
        equilSegs(i) = interp1(z, equilTemp, Zt(i));
    end

    psi_derivatives(1,i) = 1/2/pi * abs(equilTemp(floor(pos))-equilTemp(ceil(pos)))/(sqrt((Rf(i)-Ri(i))^2+(Zf(i)-Zi(i))^2)/(nPoints-1));    
    if psi_derivatives(1, i) == 0
        psi_derivatives(1, i) = 0.01;
    end
end

%% Ip matrix for XSC
dIp = -5e5;

%%%%%%%%%%%%%%%%%%%%% Ottimizzazione Nuno %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PlasmaCurrentIndex=LinearModel.PlasmaCurrentInfo.StatePosition;
% Auto Inductance for the Plasma Current
Lp=LinearModel.L(PlasmaCurrentIndex,PlasmaCurrentIndex);
% Mutual Inductance between Plasma Current and PF Coils 
Mp=LinearModel.L(PlasmaCurrentIndex,LinearModel.PoloidalCircuits.StatePosition(1:end-2));
% Solve the equation Lp.dIp/dt + Mp.dIpf/dt + Mec.dIec = 0 
% Last term is neglected - Lp.dIp/dt + Mp.dIpf/dt = 0
% dIp = -Mp/Lp.dIpf  -  Defines the influence of the PF Coils in the Plasma Current
Cp=-Mp/Lp;
% 
% GapindexOut = signalIndexByName(gapNames(fluxIndex + 1),LinearModel.OutputsInfo.Name,LinearModel.OutputsInfo.OutputPosition);
% % In the following Cg shall be the complete expression (Cg + Cgp.Cp) 
% Cg=LinearModel.C(GapindexOut,LinearModel.PoloidalCircuits.StatePosition(1:end-2))+LinearModel.C(GapindexOut,PlasmaCurrentIndex)*Cp;
% % Obtain the weights of dIpf that minimize Cg with the constraint of Plasma Current Control
% % Cp.dIpf = 1kA (or whatever current you want!)
% %  
% % x = min(Cg.dIpf-0), such that Cp.dIpf = 1000 Amp
% m=size(Cg);
% d=zeros(m(1),1);
% b=zeros(1,m(2));
% kIp = lsqlin(Cg,d,[],[],Cp, dIp);
% kIp = kIp/norm(kIp); % normalization

%%%%%%%%%%%%%%%%%%% Correnti di trasformatore %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kIp = -[7.7175
        6.6999
        6.8024
        7.6938
        0.3730
        0.4150
        4.5207
        2.9209
        0.7198
        0.2332]*1e3;
    
kIp = kIp/norm(kIp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PID controller
IpControl.Kp = 15;
IpControl.Ki = 0.5 * IpControl.Kp;
IpControl.Kd = 0;
IpControl.Td = 1;

Ip_PID = buildPID(IpControl);

% references
initialIp = Ip0;
targetIp = targetConf.LinearModel.XEquil(LinearModel.PlasmaCurrentInfo.StatePosition) + dIp;
Ip_ref = targetIp;

%% XSC setup
initialX = equilRxZx;
targetX = targetConf.LinearModel.YEquil(signalIndexByName({'CV-RX', 'CV-ZX'}, targetConf.LinearModel.OutputsInfo.Name, 1:length(targetConf.LinearModel.OutputsInfo.Name)));
X_ref = targetX;

% C matrix
XSC_C = [ones(nSeg, 1) * C_contr(signalIndexByName({'CV-FLUX'}, OutputNames, 1:length(OutputNames)), :) - C_seg; ...
         C_contr(signalIndexByName({'CV-RX', 'CV-ZX'}, OutputNames, 1:length(OutputNames)), :)];  

XSC_C = XSC_C(:, 1:nPF); % the outputs are PFC references
XSC_C = [XSC_C; Cp]; % add plasma current

% weights
wSeg = .7./psi_derivatives;
wXpos = [30 30];
wIp = .00001/Ip0;
weights_XSC = diag([wSeg, wXpos, wIp]);

[U, S, V] = svd(weights_XSC * XSC_C, 'econ'); % Singular Value Decomposition (svd(W * C))
% XSC_invC = pinv(C_XSC);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% XSC_invC = V*inv(S)*U';
NOfControlledVariables = 7;
Proj = [U(:,1:NOfControlledVariables) zeros(size(weights_XSC, 1),size(XSC_C,1)-NOfControlledVariables)]';
Proj = Proj(1:NOfControlledVariables,:);
Proj = Proj*weights_XSC;
contr2curr = V*inv(S);
contr2curr = contr2curr(:,1:NOfControlledVariables);
XSC_invC = contr2curr*Proj;

% % Remove plasma current from XSC_invC
% XSC_invC = XSC_invC(:, 1:end-1);
  
PM = XSC_invC;
VM = eye(nPF);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P = S\U'; % Projection matrix    
% % Calcolo dei valori singolari maggiori o uguali del 10% del primo
% index = 1;
% n = size(V, 1); 
% on = true;
% while(on)
%     if(index < n && S(index + 1, index + 1)>= 0.01 * S(1, 1))
%         index = index + 1;
%     else
%         on = false;
%     end
% end
% 
% PM = P(1 : index, :);
% VM = V(:, 1 : index);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% XSC PID controller
XSC_PID.Kp = .1 * ones(1, size(PM, 1));  %.1, 50
XSC_PID.Ki = 110*XSC_PID.Kp;
XSC_PID.Kd = zeros(1, nPF);
XSC_PID.Td = ones(1, nPF); % dummy value

XSC_PI = buildPID(XSC_PID);






return
%% Simulation
warning off
sim('JT_60SA_scheme_BUONO');

%% Plot util

close all

N = 500;

targetTime = [startTime, startTime+transitionTime, stopTime];
t = linspace(simGaps.time(1), simGaps.time(end), N); % equally spaced time vector

Gaps.value = interp1(simGaps.time, simGaps.signals.values, t);
RxZx.value = interp1(simGaps.time, simRxZx.signals.values, t);
Rtarget = (Rt*ones(1, N))'; % segments
Ztarget = (Zt*ones(1, N))';
RXtarget = X_ref(1)*ones(1, N); % xpoint
ZXtarget = X_ref(2)*ones(1, N);
RxZxtarget = [initialX, targetX, targetX];
% Iptarget = Ip_ref*ones(1, N);
Iptarget = [initialIp, targetIp, targetIp];
segError = interp1(simSegErr.time, simSegErr.signals.values, t);
% simGaps.time = t;

fluxMeas = interp1(simFluxMeas.time, simFluxMeas.signals.values, t);
psiX = interp1(simXFlux.time, simXFlux.signals.values, t);
segData.Ri = Ri;
segData.Rf = Rf;
segData.Zi = Zi;
segData.Zf = Zf;
segData.names = segNames;


%fig1 = figure('Position', [450, 50, 600, 150]);
fig1 = figure(1);

    plot(targetTime, Iptarget, 'r', 'LineWidth', 2)
    hold on, plot(simIp.time, simIp.signals.values, 'b')
    title('Ip')
    xlabel('Time[s]', 'Interpreter', 'latex')
    ylabel('[A]', 'Interpreter', 'latex')
    legend('ref','Ip')
    ylim([min(simIp.signals.values)-1e3, max(simIp.signals.values)+1e3]);
    xlim([0, stopTime])
%fig2 = figure('Position', [450, 250, 600, 200]);
fig2 = figure(2);

    subplot(2, 1, 1)
    plot(targetTime, RxZxtarget(1, :), 'r', 'LineWidth',2)
    hold on, plot(simRxZx.time, simRxZx.signals.values(:, 1), 'b')
    title('Rx')
    xlabel('Time[s]', 'Interpreter', 'latex')
    ylabel('[m]', 'Interpreter', 'latex')
    legend('ref','Rx')
    ylim([min(RxZxtarget(1, :))-0.01, max(RxZxtarget(1, :))+0.01]);
    xlim([0, stopTime])
    subplot(2, 1, 2)
    plot(targetTime, RxZxtarget(2, :)', 'r', 'LineWidth',2)
    hold on, plot(simRxZx.time, simRxZx.signals.values(:, 2), 'b')
    title('Zx')
    xlabel('Time[s]', 'Interpreter', 'latex')
    ylabel('[m]', 'Interpreter', 'latex')
    legend('ref','Zx')
    ylim([min(RxZxtarget(2, :))-0.01, max(RxZxtarget(2, :))+0.01]);
    xlim([0, stopTime])
%fig3 = figure('Position', [450, 500, 600, 150]);
fig3=figure(3);    
    plot(simXFlux.time, simXFlux.signals.values, 'r', 'LineWidth', 2)
    hold on
    plot(simFluxMeas.time, simFluxMeas.signals.values, 'b')
    title('Flux at the control points')
    xlabel('Time[s]', 'Interpreter', 'latex')
    ylabel('[Wb/rad]', 'Interpreter', 'latex')
    legend('X-point','control-points')
    xlim([0, stopTime])
    
% fig4 = figure(4);
% for i = 1 : nSeg
%     hold on
% %     subplot(5, 2, i)
%     plot(simSegErr.time, simSegErr.signals.values(:, i))
%     xlabel('Time[s]', 'Interpreter', 'latex')
%     ylabel('[Wb/rad]', 'Interpreter', 'latex')
%     title('Error on segments')
%     xlim([0, stopTime])
% end
% 
% fig5 = figure(5);
%     subplot(2,1,1)
%     plot(simVPFC.time, simVPFC.signals.values)
%     title('Control Voltages')
%     xlabel('Time[s]', 'Interpreter', 'latex')
%     ylabel('[V]', 'Interpreter', 'latex')
%     subplot(2,1,2)
%     plot(simPFC.time, simPFC.signals.values)
%     title('PFC currents')
%     xlabel('Time[s]', 'Interpreter', 'latex')
%     ylabel('[A]', 'Interpreter', 'latex')
%     xlim([0, stopTime])
%     
fig6 = figure(6);
    plot(interpTime,[initialBeta; initialBeta; targetBeta; targetBeta],interpTime,  [initialLi; initialLi; targetLi; targetLi])
    ylabel('$\beta_p$, $l_i$', 'Interpreter', 'latex')
    xlabel('Time[s]', 'Interpreter', 'latex')
    xlim([0, stopTime])
 



