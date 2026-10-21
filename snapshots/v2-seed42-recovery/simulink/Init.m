clc;
clear;
close all


Steer_angle=0;

Actual_Gear=3;

% TVC_on=0;%
TVC_on=0;%
%% 整车参数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%驱动防滑%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = 9.8; %重力加速度 m/s^2
veh_m0 = 3052; %整车满载质量 kg
%veh_m_real=veh_m+100; %试验质量 kg
veh_l=3.128; %轴距 m
veh_lf=1.293; %前轴到质心距离 m (满载时)
veh_lr=veh_l-veh_lf; %后轴到质心距离 m
veh_hg=0.62; %整车质心高 m
veh_Ix = 846.6; % 侧倾转动惯量 kg*m^2
veh_Iy =5757; % 俯仰转动惯量 kg*m^2
veh_Iz = 5757; % 横摆转动惯量 kg.m^2
veh_bf = 1.600; % 前轮距 m
veh_br = 1.740; % 后轮距 m
veh_b = (veh_bf+veh_br)/2;
wh_r=0.346;           %滚动半径m
wh_J = 1.877;
% %车轮参数
wh_R=0.346; %滚动半径 m
%前轴减速器参数
gear_ratio_f=9.69; %轮边减速比 
%后轴减速器参数
gear_ratio_r=9.69; %轮边减速比 
gear_efficiency=0.975;%轮边减速效率
T_motor_r = [3200	3200	3200	3200	3200	3200	3200	3200	3200	3200	3165.97005000000	2997.31125000000	2823.30435000000	2654.72055000000	2500.96635000000	2359.11780000000	2229.21970000000	2110.19050000000	2001.87100000000	1902.44225000000	1813.61550000000	1730.68675000000	1653.30485000000	1581.33130000000	1514.45380000000	1453.59245000000	1392.07110000000	1335.97365000000	1277.86485000000	1211.47635000000	1183.36360000000	1156.80015000000	1111.65720000000	1065.89000000000	1017.45000000000	969.242250000000];
w_motor_r = [51.5995872000000	103.199174400000	154.798761600000	206.398348800000	257.997936000000	309.597523200000	361.197110400000	412.796697600000	464.396284800000	515.995872000000	567.595459200000	619.195046400000	670.794633600000	722.394220800000	773.993808000000	825.593395300000	877.192982500000	928.792569700000	980.392156900000	1031.99174400000	1083.59133100000	1135.19091800000	1186.79050600000	1238.39009300000	1289.98968000000	1341.58926700000	1393.18885400000	1444.78844200000	1496.38802900000	1547.98761600000	1599.58720300000	1651.18679100000	1702.78637800000	1754.38596500000	1805.98555200000	1857.58513900000];
T_max_f=max(T_motor_r);    
T_max_r=max(T_motor_r);  
T_motor_r=T_motor_r/gear_efficiency/gear_ratio_r;
w_motor_r=w_motor_r*gear_ratio_r;
T_motor_f = T_motor_r;
w_motor_f = w_motor_r;



Cal_theta_ij=[0.3 0.3 0.3 0.3];
Cal_miu=0.3;

Cal_theta_ij=[0.8 0.8 0.8 0.8];%实车标定用
Cal_miu=0.8;%实车标定用

Cal_K_distribute_f=0.5;

% %%   Carsim输入输出接口    %%
% ips=8;            %simulink接口数量


%% 纵向车速估计
Ts=1e-3;%sampling time
gravity=9.8;% acceleration of gravity

% slope estimation 
Klomp_A=[1 -Ts*gravity
         0   1]; %state space matrix for kalman filter applicatoin
Klomp_B=[Ts
         0];  %state space matrix for kalman filter applicatoin
Klomp_C=[1 ; 0];%  坡度估计中的H矩阵  [1 0;0 1];%;  %;state space matrix for kalman filter applicatoin
Klomp_Q=[5 0
         0 0.1];%10,0.1 forTs=0.01  %process covariance matrix for slope estimation in kalman filter application
Klomp_R=200;%[200 0;0 0.1*3.14/180];%200;%200;%400[200 0;0 0.1*3.14/180] noise covariance matrix for slope estimation in kalman filter application
Klomp_slopCriti=0.3*3.14/180;% 0.01,0.1 criteria for slope error with previous one.
filter_ratio_slope=0.1; % to filter the estimated slope
slope_detec_thres=5*3.14/180;%only when the slope is larger than 5 degree, we start to judge the heading directions of vehicles(go uphill or downhill)
slope_dete_sum_thres=10;%when the uphill of downhill situation lasts more then 10 time step (0.1second), we give a conclusion of go uphill or downhill
% longitudinal velocity estimation parameters
N_num=6;
delay_length=0.05/Ts;   % delay length of wheel torque criterion
delay_constant=1/delay_length*ones(1,round(delay_length)); % this parameter is utilized in the average filter

gao_D_w = 0.03;% 0.02-ASR_DLC, 0.05-ASR, 0.3-无ASR     % 0.45; detection criterion for wheel speed change    
gao_D_w1 = 0.25;        % detection criterion for wheel acceleration and ax
gao_D_v = 0.2;%;%2，1.4;         %0.5 detection criterion for preestimation part   

excessive_wheel_intgr=1000;  %超出最大允许范围持续的时间，1000个采样时间，就是1秒
vx_max_diff=1;%四个车轮转速与四个车轮转速的平均值的差值最大允许范围
gao_mumax = 0.05;      % parameter application in wheel torque criteria which is not useful currently
gao_ay=1.5;              % 1 threshold to detect whether ay is moderate or large to use only the rear tire information
gao_gz=0.1;           % 0.05 threshold to detect whether yaw rate is moderate or large to use only the rear tire information
gao_vx=2;              %whether it's enough to utilize yawrate*vy for vx estimation
delta_thres=3*3.14/180;
deltarate_thres=0.1;
ax_vx_thres1=0.5;
ax_vx_thres2=3;
k_vx_vari_thres1=20;%50;
k_vx_vari_thres2=20;%50;
k_vx_vari_thres3=20;%50;
vx_lowest=0.01;%if estimated vx is lower than this value, the vx is set to 0
%adaptive kalman filter for vx estimation
gao_R = 0.00077574*0.1;    % 0.00077574*0.01 noise covariance matrix for vx estimation in kalman filter application 
gao_Q = 1e-02;   %5e-03 process covariance matrix for vx estimation in kalman filter application
para_long_1=1;
para_long_2=100;%100
para_long_3=100000000;%1000
para_long_4=100;%100
para_steer_1=0.1;%0.1
para_steer_2=10;%10
para_steer_3=1000;%1000
para_steer_4=1000;%1000
ax_long_1=1;
ax_long_2=3;
ax_long_3=6;
ax_steer_1=0.5;%ms^2
ax_steer_2=1.5;%ms^2
ax_steer_3=3;%ms^2
reverse_gear = 0;
vy_new_ini = 0;

%% 质心侧偏角估计
filter_ratio_ax_ay=0.2;%对ax和ay进行处理
vx_cri=1;%2 m/s if vx is slower than this value, we define vx as vx_cri
k_vy=1/veh_m0; %parameter for vy estimation
tao_1=6;      %parameter for vy estimation
k_grip_obser=10;%15 parameter for vy estimation
vy_new_ini=0;   %initial value for vy estimation
theta_ini=1;    %initial value in the vy estimation observer
time_interval=1/Ts; % how many step sizes that there is no excitation
K_theta=1;   %if there is no excitation for more than time_interval steps, the road friction will be force to converge to 1 with ratio K_theta

%excitation detection
betadot_thres=0.04;
ay_thres=0.8;
deltarate_sideslipesti_thres=0.003;
%  Kalman parameter of front Tire longitudinal force estimation
Zhang_Af=[1 -Ts*wh_r/wh_J
         0   1];
Zhang_Bf=[Ts/wh_J
         0];
Zhang_Cf=[1 ; 0];
Zhang_Qf=[1 0
         0 300000];
Zhang_Rf=2;
%  Kalman parameter of rear Tire longitudinal force estimation
Zhang_Ar=[1 -Ts*wh_r/wh_J
         0   1];
Zhang_Br=[Ts/wh_J
         0];
Zhang_Cr=[1 ; 0];
Zhang_Qr=[1 0
         0 300000];
Zhang_Rr=2;
% EKF for tire road friction based on lateral dynamics
Q_EKF_lat=[(0.1)^2 0 0;0 (0.1)^2 0;0 0 (0.1)^2];
R_EKF_lat=[0.01^2 0 0;0 0.1^2 0;0 0 0.05^2];
deg_EKF_thres=2;%
vx_mulat_lowest=vx_cri+0.5;

%% 路面估计
slip_lowthre = 0.035; %纵向估计器启停滑动率下限
slip_upthre = 0.2; %纵向估计器启停滑动率上限
slip_thre = 0.05;
% slip_upthre = 0.3;
alpha_thre = 0.03; %侧向估计器启停轮胎侧偏角下限
alpha_upthre = 0.1; %侧向估计器启停轮胎侧偏角上限
u_init = 0.9;
K_Fx_est = 20; % 20; 纵向力估计器增益
Kg = 15;%路面附着系数估计器增益
Kdyn = -0.03;
vx_thre = 22;
vx_upthre = 200;
miu_thre = 0.005;
slip_timelim = 0.1;
slip_thre_xy = 0.03; 
alpha_thre_xy = 0.045; 

BF = [34.74,-0.6056,0.2956,0.1383];
Brush = [28.93,1.496,0.9486,0.06054];
slip_re = [0,0.02,0.03,0.04,0.06,0.07,0.09,0.10,0.12,0.13,0.15,0.17];
theta_slip = 0:0.1:1.1;
delta_x_max = 0.06;
delta_y_max = 0.07;
%%
theta2=34.74;
theta3=-0.6056;
theta4=0.2956;
theta5=0.1383;
muf=@(x,lambda) x-x*exp(-theta2/x*(lambda+theta3*lambda^2))-theta4*lambda*sign(lambda)+theta5*lambda^2;
slip=@(x,lambda) -x*exp(-theta2/x*(lambda+theta3*lambda^2))*(-theta2/x*(1+2*theta3*lambda))-theta4+2*theta5*lambda;

k=1;
x=0;
for x=0:0.1:1.1
    i=1;
    for lambda=0:0.01:0.4 
        u(i)=lambda;
        mu(i)=muf(x,lambda);
        slip_r(i)=slip(x,lambda);
        
        if i==1
            j=1;
        else
         if slip_r(i)*slip_r(i-1)<=0
             if abs(slip_r(i))<abs(slip_r(i-1))
                 j=i;
             else
                 j=i-1;
             end
         end
        end
    
        i=i+1;  
    end
    x1(k)=x;
%     [s,l]=min(abs(slip_r));
%     m(
    slip_re(k)=u(j);
    k=k+1;
%     plot(u,mu);
%     hold on
end

Cal_Tmax_rate_lim_add=100;
Cal_Tmax_rate_lim_sub=100;

%% 单位转化常数 %%
dg2rd = pi/180;
rd2dg = 180/pi;
rpm2rad = 2*pi/60;

%% 坡度估计
Ts = 0.001;
Zhang_Af=[1 -Ts*wh_R/wh_J
         0   1];
Zhang_Bf=[Ts/wh_J
         0];
Zhang_Cf=[1 ; 0];
Zhang_Qf=[1 0
         0 300000];
Zhang_Rf=2;
Klomp_A=[1 -Ts*g
         0   1]; %state space matrix for kalman filter applicatoin
Klomp_B=[Ts
         0];  %state space matrix for kalman filter applicatoin
Klomp_C=[1 ; 0];%  坡度估计中的H矩阵  [1 0;0 1];%;  %;state space matrix for kalman filter applicatoin
Klomp_Q=[5 0
         0 0.1];%10,0.1 forTs=0.01  %process covariance matrix for slope estimation in kalman filter application
Klomp_R=200;
Klomp_slopCriti=0.3*3.14/180;
slope_lambda = 0.1;%0.97
