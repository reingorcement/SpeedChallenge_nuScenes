function [v, v_best_believe] = fcn(vrc, ews, ay, gao_ay, gz, gao_gz, Fxfl_sign, Fxfr_sign, Fxrl_sign, Fxrr_sign, delta_rad, deltarate_rads, delta_thres, deltarate_thres)

    vFL = vrc(1); % v_corr FL
    vFR = vrc(2); % v_corr FR
    vRL = vrc(3); % v_corr RL
    vRR = vrc(4); % v_corr RR
    ewsFL = ews(1); % ews FL
    ewsFR = ews(2); % ews FR
    ewsRL = ews(3); % ews RL
    ewsRR = ews(4); % ews RR
    v = zeros(1, 1);
    v_choose_p = zeros(4, 2);
    v_choose_l = zeros(4, 2);
    v_choose = [ewsFL vFL Fxfl_sign;
                ewsFR vFR Fxfr_sign;
                ewsRL vRL Fxrl_sign;
                ewsRR vRR Fxrr_sign]; % All velocities with corresponding EWS Flag
    v_best_believe = boolean(1);

    %我们发现，纵向加速度的测量值会受到坡度的影响，因此有时纵向加速度为正，实际车辆在减速，容易影响最优车速的选择；此外横摆角速度和侧向车速也会影响纵向加速度。因此我们采用每个车轮上是否有加速或减速意图（通过轮胎上施加的是驱动力还是制动力来判断）来判断某个轮胎处于驱动还是制动状态。
    if ((abs(ay) >= gao_ay | abs(gz) > gao_gz | abs(delta_rad) > delta_thres | abs(deltarate_rads) > deltarate_thres))
        % 有侧向运动的情况
        if sum(ews(1:4)) == 4 % If all wheels have EWS, choose from all 4 wheels
            v_choose_f1 = [vFL Fxfl_sign; vFR Fxfr_sign; vRL Fxrl_sign; vRR Fxrr_sign];
            v = all_wheel_slip(v_choose_f1(:, 1), sum(v_choose_f1(:, 2)));
        else
            lenth_l = sum(v_choose(:, 1) == 0); % 不打滑车轮数量
            num_index_l = (v_choose(:, 1) == 0); % 车轮打滑为0，不打滑为1
            jj = 0;

            for ii = 1:4

                if num_index_l(ii) == 1
                    jj = jj + 1;
                    v_choose_l(jj, :) = v_choose(ii, 2:3); % 筛选不打滑车轮的信息
                end

            end

            if lenth_l == 4 % 四轮不打滑
                v = all_wheel_slip_lat(v_choose_l(1:lenth_l, 1), lenth_l);
            elseif lenth_l == 3 % 三轮不打滑
                v = all_wheel_slip_lat(v_choose_l(1:lenth_l, 1), lenth_l);
            else
                v = mean(v_choose_l(1:lenth_l, 1));
            end

        end

        v_best_believe = boolean(1);
    else

        if sum(ews(1:4)) == 4 % If all wheels have EWS, choose from all 4 wheels
            v_choose_f2 = [vFL Fxfl_sign; vFR Fxfr_sign; vRL Fxrl_sign; vRR Fxrr_sign];
            v = all_wheel_slip(v_choose_f2(:, 1), sum(v_choose_f2(:, 2))); %取最小或最大轮速
        else
            lenth = sum(v_choose(:, 1) == 0);
            num_index = (v_choose(:, 1) == 0);
            j = 0;

            for i = 1:4

                if num_index(i) == 1
                    j = j + 1;
                    v_choose_p(j, :) = v_choose(i, 2:3); % 筛选不打滑车轮的信息
                end

            end

            if (sum(v_choose_p(:, 2) > 0) == lenth) % 不打滑车轮的纵向力均为正
                v = min(v_choose_p(1:lenth, 1));
            elseif (sum(v_choose_p(:, 2) < 0) == lenth) % 不打滑车轮的纵向力均为负
                v = max(v_choose_p(1:lenth, 1));
            else % 不打滑车轮的纵向力正负不统一
                v = all_wheel_slip_long(v_choose_p(:, 1), lenth);
            end

        end

        v_best_believe = boolean(0);
    end

end
