% quadsim_control.m
%
% Flight control logic for quadsim
%
% Inputs:
%   Trajectory commands
%   State Feedbacks
%   Time
%
% Outputs:
%   Control surface commands
%   Autopilot state commands (for logging and plotting)
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   
function out = quadsim_control(uu,P)

    % Extract variables from input vector uu
    %   uu = [traj_cmds(1:4); estimates(1:23); time(1)];
    k=(1:4);         traj_cmds=uu(k); % Trajectory Commands
    k=k(end)+(1:23); estimates=uu(k); % Feedback state estimates
    k=k(end)+(1);    time=uu(k);      % Simulation time, s

    % Extract variables from traj_cmds
    h_c      = traj_cmds(1);  % commanded altitude (m)
    Vhorz_c  = traj_cmds(2);  % commanded horizontal speed (m/s) (change from uavsim)
    chi_c    = traj_cmds(3);  % commanded course (rad)
    psi_c    = traj_cmds(4);  % yaw course (rad) (change from uavsim)

    % Extract variables from estimates
    pn_hat       = estimates(1);  % inertial North position, m
    pe_hat       = estimates(2);  % inertial East position, m
    h_hat        = estimates(3);  % altitude, m
    Va_hat       = estimates(4);  % airspeed, m/s
    phi_hat      = estimates(5);  % roll angle, rad
    theta_hat    = estimates(6);  % pitch angle, rad
    psi_hat      = estimates(7);  % yaw angle, rad
    p_hat        = estimates(8);  % body frame roll rate, rad/s
    q_hat        = estimates(9);  % body frame pitch rate, rad/s
    r_hat        = estimates(10); % body frame yaw rate, rad/s
    Vn_hat       = estimates(11); % north speed, m/s
    Ve_hat       = estimates(12); % east speed, m/s
    Vd_hat       = estimates(13); % downward speed, m/s
    wn_hat       = estimates(14); % wind North, m/s
    we_hat       = estimates(15); % wind East, m/s    
    future_use   = estimates(16:23);

    % Initialize controls to trim (to be with PID logic)
    delta_e=P.delta_e0;
    delta_a=P.delta_a0;
    delta_r=P.delta_r0;
    delta_t=P.delta_t0;

    % Set "first-time" flag, which is used to initialize PID integrators
    firstTime=(time==0);
    
    % Initialize autopilot commands (may be overwritten with PID logic)
    [WP_n, WP_e, h_c, psi_c] = get_quadsim_trajectory_commands(time);
    R_ned2b = eulerToRotationMatrix(phi_hat, theta_hat, psi_hat);
    k_pos = .1; % gain
    chi_c = atan2(WP_e - pe_hat, WP_n - pn_hat);
    Vhorz_c = k_pos*sqrt((WP_e - pe_hat)^2 + (WP_n - pn_hat)^2);
    Vhorz_c = min(Vhorz_c, P.Vhorz_max);
    Vn_c = Vhorz_c*cos(chi_c);
    Ve_c = Vhorz_c*sin(chi_c);
    Vhorz_xyc = [
        cos(psi_hat) sin(psi_hat); 
        -sin(psi_hat) cos(psi_hat)
        ] * [
        Vn_c; Ve_c
        ];
    Vhorz_xc = Vhorz_xyc(1);
    Vhorz_yc = Vhorz_xyc(2);
    Vhorz_xy = [
        cos(psi_hat) sin(psi_hat); 
        -sin(psi_hat) cos(psi_hat)
        ] * [
        Vn_hat; Ve_hat
        ];
    Vhorz_x = Vhorz_xy(1);
    Vhorz_y = Vhorz_xy(2);

    %% Flight control logic
    % Note: For logging purposes, use variables: 
    %         Vhorz_c,  chi_c, h_c, phi_c, theta_c, psi_c
    %% Flight control logic
    %% Note: may use one of the following methods to tune:
    %%  1) tuning process proposed by Beard & Mcclain.
    %%  2) tune via PIR controller

    %% tune course controller
    rn_hat = Vn_hat;
    re_hat = Ve_hat;
    chi_hat = atan2(re_hat,rn_hat);

    %% tune roll controller
    P.roll_kp = 0.01;
    P.roll_ki = 0.002;
    P.roll_kd = 0.006;

    %% tune altitude controller
    P.altitude_kp = 0.07;
    P.altitude_ki = 0.015;
    P.altitude_kd = 0.08; % 0.06

    %% tune pitch controller
    P.pitch_kp = 0.012; % 0.01
    P.pitch_ki = 0.002;
    P.pitch_kd = 0.01;
   
    %% tune yaw controller
    P.yaw_kp = 0.1; % 0.2
    P.yaw_ki = 0.012;
    P.yaw_kd = 0.1;
    
    %% tune velocity x controller
    P.vhx_kp = .1; 
    P.vhx_ki = .05;
    P.vhx_kd = 0.0;

    %% tune velocity y controller
    P.vhy_kp = .1;
    P.vhy_ki = .05;
    P.vhy_kd = 0.0;

    persistent h_prev
    if(firstTime)
        h_prev = P.h0_m;
        % Initialize integrators
        PIR_pitch_hold(0,0,0,firstTime,P); 
        PIR_alt_hold_using_throttle(0,0,0,firstTime,P); 
        PIR_roll_hold(0, 0, 0, firstTime, P);
        PIR_yaw_hold(0, 0, 0, firstTime, P);
    end

    h_rate = -(h_prev-h_hat)/P.Ts;
    theta_c = -PIR_Vhorz_x_hold(Vhorz_xc, Vhorz_x, 0, firstTime, P);
    phi_c = PIR_Vhorz_y_hold(Vhorz_yc, Vhorz_y, 0, firstTime, P);
    delta_e = PIR_pitch_hold(theta_c, theta_hat, q_hat, firstTime, P);
    delta_a = PIR_roll_hold(phi_c, phi_hat, p_hat, firstTime, P);
    delta_r = PIR_yaw_hold(psi_c, psi_hat, r_hat, firstTime, P);
    delta_t = PIR_alt_hold_using_throttle(h_c, h_hat, h_rate, firstTime, P);
    h_prev = h_hat;


    % Compile vector of control surface deflections
    delta = [ ...
            delta_e; ...
            delta_a; ...
            delta_r; ...
            delta_t; ...
        ];

    % Override control delta with manual flight delta
    if P.manual_flight_flag
        error('Manual flight not supported in quadsim')
    end

    % Compile autopilot commands for logging/vis
    ap_command = [ ...
            Vhorz_c; ...
            h_c; ...
            chi_c; ...
            phi_c; ...
            theta_c; 
            psi_c; ... % change from uavsim
            0; ... % future use
            0; ... % future use
            0; ... % future use
        ];

    % Compile output vector
    out=[delta;ap_command]; % 4+9=13

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Autopilot functions
%
%   Autopilot controllers in UAVSIM are based on "PI with rate feedback".
%   For convenience, we'll refer to "PI with rate feedback" as "PIR".
%
%   u = PIR_xxx(y_c, y, y_dot, init_flag, dt)
%     Inputs:
%       y_c:    Closed loop command
%       y:      Current system response
%       y_dot:  Rate feedback (derivative of y)
%       init_flag:  1: initialize integrator, 0: otherwise
%       dt:     Time step, seconds
%     Outputs:
%       u:      Controller output (input to Gplant)
%                         
%                .------.           .---- Limit on plant input
%             .->| ki/s |---.       |     (a.k.a. limit on controller output)
%             |  '------'   |+      v
%   Input     |  .------. + v  +    -. u  .------. .---.     .---.  Output
%    ---->( )-'->|  kp  |->( )->( )--|--->|Gplant|-| s |--.--|1/s|--.--->
%   y_c    ^     '------'        ^  -'    '------' '---'  |  '---'  |  y
%         -|                    -|         .------.       |         |
%          |                     '---------|  kd  |<------'y_dot    |
%          |                               '------'                 |
%          '--------------------------------------------------------'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vh_x
%   - regulate east component of velocity using roll
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_Vhorz_x_hold(Vhorz_xc, Vhorz_x_hat, q_hat, init_flag, P)

    % Set up PI with rate feedback
    y_c = Vhorz_xc; % Command
    y = Vhorz_x_hat; % Feedback
    y_dot = q_hat; % Rate feedback
    kp = P.vhx_kp;
    ki = P.vhx_ki;
    kd = P.vhx_kd;
    u_lower_limit = -P.theta_max;
    u_upper_limit = +P.theta_max;
%     u_lower_limit = -P.phi_max;
%     u_upper_limit = +P.phi_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
    error = y_c - y;  % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vh_y
%   - regulate north component of velocity using pitch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_Vhorz_y_hold(Vhorz_yc, Vhorz_y_hat, q_hat, init_flag, P)

    % Set up PI with rate feedback
    y_c = Vhorz_yc; % Command
    y = Vhorz_y_hat; % Feedback
    y_dot = q_hat; % Rate feedback
    kp = P.vhy_kp;
    ki = P.vhy_ki;
    kd = P.vhy_kd;
    u_lower_limit = -P.phi_max;
    u_upper_limit = +P.phi_max;
%     u_lower_limit = -P.phi_max;
%     u_upper_limit = +P.phi_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
    error = y_c - y;  % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pitch_hold
%   - regulate pitch using elevator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_pitch_hold(theta_c, theta_hat, q_hat, init_flag, P)

    % Set up PI with rate feedback
    y_c = theta_c; % Command
    y = theta_hat; % Feedback
    y_dot = q_hat; % Rate feedback
    kp = P.pitch_kp;
    ki = P.pitch_ki;
    kd = P.pitch_kd;
    u_lower_limit = -P.delta_e_max;
    u_upper_limit = +P.delta_e_max;
%     u_lower_limit = -P.phi_max;
%     u_upper_limit = +P.phi_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
    error = y_c - y;  % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int - kd*y_dot;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alt_hold
%   - regulate altitude using pitch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_alt_hold_using_throttle(h_c, h_hat, h_rate, init_flag, P)

    % Set up PI with rate feedback
    y_c = h_c;   % Command
    y = h_hat;   % Feedback
    y_dot = h_rate;   % Rate feedback
    kp = P.altitude_kp;
    ki = P.altitude_ki;
    kd = P.altitude_kd;
%     u_lower_limit = -P.theta_max / P.K_theta_DC;
%     u_upper_limit = +P.theta_max / P.K_theta_DC;
    u_lower_limit = P.delta_t_min;
    u_upper_limit = P.delta_t_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
%     error = y_c - y;  % Error between command and response
    if y_c - y < 0
        error = max(y_c - y, -10); % Error between command and response
    else
        error = min(y_c - y, 10); % Error between command and response
    end
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int - kd*y_dot;
%     u = u + 0.5;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alt_hold
%   - regulate altitude using pitch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_alt_hold_using_pitch(h_c, h_hat, q_hat, init_flag, P)

    % Set up PI with rate feedback
    y_c = h_c;   % Command
    y = h_hat;   % Feedback
    y_dot = q_hat;   % Rate feedback
    kp = P.altitude_kp;
    ki = P.altitude_ki;
    kd = P.altitude_kd;
%     u_lower_limit = -P.theta_max / P.K_theta_DC;
%     u_upper_limit = +P.theta_max / P.K_theta_DC;
    u_lower_limit = -P.theta_max;
    u_upper_limit = +P.theta_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
    error = y_c - y;  % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int - kd*y_dot;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% roll
%   - regulate roll using elevator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_roll_hold(phi_c, phi_hat, p_hat, firstTime, P)
    % Set up PI with rate feedback
    y_c = phi_c; % Command
    y = phi_hat; % Feedback
    y_dot = p_hat; % Rate feedback
    kp = P.roll_kp;
    ki = P.roll_ki;
    kd = P.roll_kd;
    u_lower_limit = -P.delta_a_max;
    u_upper_limit = +P.delta_a_max;
%     u_lower_limit = -P.phi_max;
%     u_upper_limit = +P.phi_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( firstTime )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
    error = y_c - y;  % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int - kd*y_dot;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% yaw
%   - regulate yaw using rudder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_yaw_hold(psi_c, psi_hat, r_hat, init_flag, P)

%     if (psi_c > pi)
%         psi_c = -((2*pi) - psi_c);
%     elseif (psi_c < -pi)
%         psi_c = (2*pi) + psi_c;
%     end
%     if (psi_hat > pi)
%         psi_hat = -((2*pi) - psi_hat);
%     elseif (psi_hat < -pi)
%         psi_hat = (2*pi) + psi_hat;
%     end

    % Set up PI with rate feedback
    y_c = psi_c; % Command
    y = psi_hat; % Feedback
    y_dot = r_hat; % Rate feedback
    kp = P.yaw_kp;
    ki = P.yaw_ki;
    kd = P.yaw_kd;
    u_lower_limit = -P.delta_r_max;
    u_upper_limit = +P.delta_r_max;

    % Initialize integrator (e.g. when t==0)
    persistent error_int;
    if( init_flag )   
        error_int = 0;
    end  

    % Perform "PI with rate feedback"
%     error = y_c - y;  % Error between command and response
    error = mod(y_c-y + pi, 2*pi) -pi; % Error between command and response
    error_int = error_int + P.Ts*error; % Update integrator
    u = kp*error + ki*error_int - kd*y_dot;

    % Output saturation & integrator clamping
    %   - Limit u to u_upper_limit & u_lower_limit
    %   - Clamp if error is driving u past limit
    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error>0
            error_int = error_int - P.Ts*error;
        end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error<0
            error_int = error_int - P.Ts*error;
        end
    end
end
