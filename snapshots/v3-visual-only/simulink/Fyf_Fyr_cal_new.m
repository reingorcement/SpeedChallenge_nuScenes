function [Fy_f,Fy_r,pderi_Fyf_vy,pderi_Fyr_vy,alpha_f,alpha_r] = Fyf_Fyr_cal_new(theta,lf,lr,bf,br,a,b,c,d,vx,yawrate,delta,vy,Fz_fl,Fz_fr,Fz_rl,Fz_rr)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
% alpha_f=(vy+omega*lf)/vx-delta;
% alpha_r=(vy-omega*lr)/vx;
beta = (vy/max(0.1,vx));
% alpha_fl = double((vx*beta+lf*yawrate)/(vx-bf*yawrate/2)-delta)+sign(yawrate)*(min(0.04,0.05-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
% alpha_fr = double((vx*beta+lf*yawrate)/(vx+bf*yawrate/2)-delta)+sign(yawrate)*(min(0.04,0.05-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
% alpha_rl = double(1.25*(beta-lr*yawrate/(vx+br*yawrate/2)))+sign(yawrate)*(min(0.05,0.06-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
% alpha_rr = double(1.25*(beta-lr*yawrate/(vx-br*yawrate/2)))+sign(yawrate)*(min(0.05,0.06-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
alpha_fl0 = double((vx*beta+lf*yawrate)/(vx-bf*yawrate/2)-delta); % +sign(yawrate)*(min(0.04,0.05-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
alpha_fr0 = double((vx*beta+lf*yawrate)/(vx+bf*yawrate/2)-delta); % +sign(yawrate)*(min(0.04,0.05-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
alpha_rl0 = double((beta-lr*yawrate/(vx+br*yawrate/2))); % +sign(yawrate)*(min(0.05,0.06-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
alpha_rr0 = double((beta-lr*yawrate/(vx-br*yawrate/2))); % +sign(yawrate)*(min(0.05,0.06-vx*0.0016)/2)*min(abs(delta)*180/pi*0.5,1);
alpha_fl = min(max(0.05+5*abs(alpha_fl0+alpha_fr0),0.6),0.95)*alpha_fl0;
alpha_fr = min(max(0.05+5*abs(alpha_fl0+alpha_fr0),0.6),0.95)*alpha_fr0;
alpha_rl = min(max(0.05+6.5*abs(alpha_rl0+alpha_rr0),0.6),1.0)*alpha_rl0;
alpha_rr = min(max(0.05+6.5*abs(alpha_rl0+alpha_rr0),0.6),1.0)*alpha_rr0;
% alpha_fl = double((vx*beta+lf*yawrate)/(vx-bf*yawrate/2)-delta);
% alpha_fr = double((vx*beta+lf*yawrate)/(vx+bf*yawrate/2)-delta);
alpha_f = (alpha_fl+alpha_fr);
% alpha_rl = double(1.2*(beta-lr*yawrate/(vx+br*yawrate/2)));
% alpha_rr = double(1.2*(beta-lr*yawrate/(vx-br*yawrate/2)));
alpha_r = (alpha_rl+alpha_rr);
% theta=1;% 默认的路面附着系数
Fz_f = (Fz_fl+Fz_fr)/2;
Fz_r = (Fz_rl+Fz_rr)/2;

Fy_fl = -sign(alpha_fl)*((b*Fz_fl^(c-d)*a*abs(alpha_fl)*(1-a*abs(alpha_fl)/(3*theta*Fz_fl^d)+1/3*(a*abs(alpha_fl)/(3*theta*Fz_fl^d))^2))*(abs(alpha_fl)<3*theta*Fz_fl^d/a)+b*theta*Fz_fl^c*(abs(alpha_fl)>=3*theta*Fz_fl^d/a));
Fy_fr = -sign(alpha_fr)*((b*Fz_fr^(c-d)*a*abs(alpha_fr)*(1-a*abs(alpha_fr)/(3*theta*Fz_fr^d)+1/3*(a*abs(alpha_fr)/(3*theta*Fz_fr^d))^2))*(abs(alpha_fr)<3*theta*Fz_fr^d/a)+b*theta*Fz_fr^c*(abs(alpha_fr)>=3*theta*Fz_fr^d/a));
Fy_rl = -sign(alpha_rl)*((b*Fz_rl^(c-d)*a*abs(alpha_rl)*(1-a*abs(alpha_rl)/(3*theta*Fz_rl^d)+1/3*(a*abs(alpha_rl)/(3*theta*Fz_rl^d))^2))*(abs(alpha_rl)<3*theta*Fz_rl^d/a)+b*theta*Fz_rl^c*(abs(alpha_rl)>=3*theta*Fz_rl^d/a));
Fy_rr = -sign(alpha_rr)*((b*Fz_rr^(c-d)*a*abs(alpha_rr)*(1-a*abs(alpha_rr)/(3*theta*Fz_rr^d)+1/3*(a*abs(alpha_rr)/(3*theta*Fz_rr^d))^2))*(abs(alpha_rr)<3*theta*Fz_rr^d/a)+b*theta*Fz_rr^c*(abs(alpha_rr)>=3*theta*Fz_rr^d/a));
Fy_f=Fy_fl+Fy_fr;
Fy_r=Fy_rl+Fy_rr;
% Fy_f = -2*sign(alpha_f)*((b*Fz_f^(c-d)*a*abs(alpha_f)*(1-a*abs(alpha_f)/(3*theta*Fz_f^d)+1/3*(a*abs(alpha_f)/(3*theta*Fz_f^d))^2))*(abs(alpha_f)<3*theta*Fz_f^d/a)+b*theta*Fz_f^c*(abs(alpha_f)>=3*theta*Fz_f^d/a));
% Fy_r = -2*sign(alpha_r)*((b*Fz_r^(c-d)*a*abs(alpha_r)*(1-a*abs(alpha_r)/(3*theta*Fz_r^d)+1/3*(a*abs(alpha_r)/(3*theta*Fz_r^d))^2))*(abs(alpha_r)<3*theta*Fz_r^d/a)+b*theta*Fz_r^c*(abs(alpha_r)>=3*theta*Fz_r^d/a));


%接下来计算各项偏导数
if(alpha_fl>=0)
    pderi_Fy_alpha_fl=-((Fz_fl^(c-3*d)*a*b*(a*alpha_fl-3*Fz_fl^d*theta)^2)/(9*theta^2))*(abs(alpha_fl)<3*theta*Fz_fl^d/a);
else
    pderi_Fy_alpha_fl=(-(Fz_fl^(c-3*d)*a*b*(a*alpha_fl+3*Fz_fl^d*theta)^2)/(9*theta^2))*(abs(alpha_fl)<3*theta*Fz_fl^d/a);
end
if(alpha_fr>=0)
    pderi_Fy_alpha_fr=-((Fz_fr^(c-3*d)*a*b*(a*alpha_fr-3*Fz_fr^d*theta)^2)/(9*theta^2))*(abs(alpha_fr)<3*theta*Fz_fr^d/a);
else
    pderi_Fy_alpha_fr=(-(Fz_fr^(c-3*d)*a*b*(a*alpha_fr+3*Fz_fr^d*theta)^2)/(9*theta^2))*(abs(alpha_fr)<3*theta*Fz_fr^d/a);
end
if(alpha_rl>=0)
    pderi_Fy_alpha_rl=-((Fz_rl^(c-3*d)*a*b*(a*alpha_rl-3*Fz_rl^d*theta)^2)/(9*theta^2))*(abs(alpha_rl)<3*theta*Fz_rl^d/a);
else
    pderi_Fy_alpha_rl=(-(Fz_rl^(c-3*d)*a*b*(a*alpha_rl+3*Fz_rl^d*theta)^2)/(9*theta^2))*(abs(alpha_rl)<3*theta*Fz_rl^d/a);
end
if(alpha_rr>=0)
    pderi_Fy_alpha_rr=-((Fz_rr^(c-3*d)*a*b*(a*alpha_rr-3*Fz_rr^d*theta)^2)/(9*theta^2))*(abs(alpha_rr)<3*theta*Fz_rr^d/a);
else
    pderi_Fy_alpha_rr=(-(Fz_rr^(c-3*d)*a*b*(a*alpha_rr+3*Fz_rr^d*theta)^2)/(9*theta^2))*(abs(alpha_rr)<3*theta*Fz_rr^d/a);
end

pderi_Fy_alpha_f=pderi_Fy_alpha_fl+pderi_Fy_alpha_fr;
pderi_Fy_alpha_r=pderi_Fy_alpha_rl+pderi_Fy_alpha_rr;

pderi_alpha_f_vy=1/vx;
pderi_Fyf_vy=pderi_Fy_alpha_f*pderi_alpha_f_vy;

pderi_alpha_r_vy=1/vx;
pderi_Fyr_vy=pderi_Fy_alpha_r*pderi_alpha_r_vy;
end

