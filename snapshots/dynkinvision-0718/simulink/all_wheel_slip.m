function v= all_wheel_slip(u,Fx_sign)
%ALL_WHEEL_SLIP 此处显示有关此函数的摘要
%   此处显示详细说明

if Fx_sign<=0
    v=max(u);
else
    v=min(u);
end

% num=length(u);
% dis_1=abs(sum(u)/4-u(1));
% dis_2=abs(sum(u)/4-u(2));
% dis_3=abs(sum(u)/4-u(3));
% dis_4=abs(sum(u)/4-u(4));
% dis=[dis_1;dis_2;dis_3;dis_4];
% [M,I]=max(dis);
% if I>=1 & I<=2
% v=(u(3)+u(4))/2;
% else
% v=(u(1)+u(2))/2;  
% end
end

