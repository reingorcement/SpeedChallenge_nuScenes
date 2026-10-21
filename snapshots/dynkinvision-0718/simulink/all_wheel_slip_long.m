function v= all_wheel_slip_long(u,lenth)
%ALL_WHEEL_SLIP 此处显示有关此函数的摘要
%   此处显示详细说明
num=lenth;
if num==4  % 4轮不打滑
    dis_1=abs(sum(u)/4-u(1));
    dis_2=abs(sum(u)/4-u(2));
    dis_3=abs(sum(u)/4-u(3));
    dis_4=abs(sum(u)/4-u(4));
elseif num==3  % 3轮不打滑
    dis_1=abs(sum(u)/3-u(1));
    dis_2=abs(sum(u)/3-u(2));
    dis_3=abs(sum(u)/3-u(3));
    dis_4=abs(sum(u)/3);
else  % 2或1抡不打滑
    dis_1=abs(sum(u)/2-u(1));
    dis_2=abs(sum(u)/2-u(2));
    dis_3=abs(sum(u)/2);
    dis_4=abs(sum(u)/2);
end

dis=[dis_1;dis_2;dis_3;dis_4];
[M,I]=max(dis);
if I>=1 & I<=2 %如果这个情况出现，说明4个车轮都不打滑，那么就取u(3)和u（4）的平均值
v=(u(3)+u(4))/2;
else   %因为u(1)和u(2)都是不打滑的，那么这种情况下就选择u(1)和u(2)的平均值
v=(u(1)+u(2))/2;  
end
end

