function v= all_wheel_slip_lat(u,lenth_l)
%ALL_WHEEL_SLIP 此处显示有关此函数的摘要
%   此处显示详细说明
if lenth_l==3  % 3个车轮不打滑，取2个与平均值最近的车轮轮速的均值为v
    v_ave=mean(u);
    dis_1=abs(v_ave-u(1));
    dis_2=abs(v_ave-u(2));
    dis_3=abs(v_ave-u(3));
    dis=[dis_1;dis_2;dis_3];
    [M,I]=max(dis);
    if I==1
        v=(u(2)+u(3))/2;
    elseif I==2
        v=(u(1)+u(3))/2;
    else
        v=(u(1)+u(2))/2;
    end
else % 四轮均不打滑，取3个与平均值最近的车轮轮速的均值为v
    v_ave=mean(u);
    dis_1=abs(v_ave-u(1));
    dis_2=abs(v_ave-u(2));
    dis_3=abs(v_ave-u(3));
    dis_4=abs(v_ave-u(4));
    dis=[dis_1;dis_2;dis_3;dis_4];
    [MM,II]=max(dis);
    if II==1
        v=(u(2)+u(3)+u(4))/3;
    elseif II==2
        v=(u(1)+u(3)+u(4))/3;
    elseif II==3
        v=(u(1)+u(2)+u(4))/3;
    else
        v=(u(1)+u(2)+u(3))/3;
    end
end
end
