% quadsim_control.m
%
% Closed-loop flight controller for the quadrotor simulator.
%
% Implements a cascaded PI-with-rate-feedback (PIR) autopilot that tracks
% altitude, horizontal velocity, heading, roll, pitch, and yaw commands.
%
% The outer loop converts trajectory commands into attitude/rate setpoints;
% the inner loop closes the loop using body-rate feedback from the estimator.
%
% Inputs (flat uu vector, Simulink S-function convention):
%   uu = [traj_cmds(1:4); estimates(1:23); time(1)]
%
%   traj_cmds  - [h_c, Vhorz_c, chi_c, psi_c]: altitude (m), horizontal
%                speed (m/s), course (rad), and yaw (rad) commands
%   estimates  - 23-element state estimate vector from quadsim_estimates
%   time       - simulation time, s
%
% Outputs:
%   out = [delta(1:4); ap_command(1:9)]
%   delta      - [delta_e, delta_a, delta_r, delta_t]: control channel outputs
%   ap_command - commanded states logged for plotting: [Vhorz_c, h_c, chi_c,
%                phi_c, theta_c, psi_c, 0, 0, 0]
%
% Controller gains are set directly in this file (see "Controller gains" below).
% To change the commanded trajectory, edit get_quadsim_trajectory_commands.m.
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012
function out = quadsim_control(uu, P)

    % Unpack input vector
    %   uu = [traj_cmds(1:4); estimates(1:23); time(1)]
    k=(1:4);         traj_cmds=uu(k); % Trajectory commands
    k=k(end)+(1:23); estimates=uu(k); % State estimates
    k=k(end)+(1);    time=uu(k);      % Simulation time, s

    % Unpack trajectory commands
    h_c      = traj_cmds(1);  % commanded altitude, m
    Vhorz_c  = traj_cmds(2);  % commanded horizontal speed, m/s
    chi_c    = traj_cmds(3);  % commanded course, rad
    psi_c    = traj_cmds(4);  % commanded yaw, rad

    % Unpack state estimates
    pn_hat     = estimates(1);   % inertial North position, m
    pe_hat     = estimates(2);   % inertial East position, m
    h_hat      = estimates(3);   % altitude, m
    Va_hat     = estimates(4);   % airspeed, m/s
    phi_hat    = estimates(5);   % roll angle, rad
    theta_hat  = estimates(6);   % pitch angle, rad
    psi_hat    = estimates(7);   % yaw angle, rad
    p_hat      = estimates(8);   % body roll rate, rad/s
    q_hat      = estimates(9);   % body pitch rate, rad/s
    r_hat      = estimates(10);  % body yaw rate, rad/s
    Vn_hat     = estimates(11);  % north speed, m/s
    Ve_hat     = estimates(12);  % east speed, m/s
    Vd_hat     = estimates(13);  % downward speed, m/s
    wn_hat     = estimates(14);  % wind North, m/s
    we_hat     = estimates(15);  % wind East, m/s
    future_use = estimates(16:23);

    % Initialize controls to trim (overwritten below by PIR logic)
    delta_e = P.delta_e0;
    delta_a = P.delta_a0;
    delta_r = P.delta_r0;
    delta_t = P.delta_t0;

    % "First time" flag — resets persistent integrators at t=0
    firstTime = (time == 0);

    % --- Waypoint navigation ---
    % Look up the current waypoint from the mission table and convert it
    % into body-frame velocity commands to drive the vehicle toward the waypoint.
    [WP_n, WP_e, h_c, psi_c] = get_quadsim_trajectory_commands(time);
    k_pos   = 0.1;  % position-to-speed gain (1/s): scales distance error to speed command
    chi_c   = atan2(WP_e - pe_hat, WP_n - pn_hat);          % bearing to waypoint, rad
    Vhorz_c = k_pos * sqrt((WP_e - pe_hat)^2 + (WP_n - pn_hat)^2);
    Vhorz_c = min(Vhorz_c, P.Vhorz_max);                    % cap speed at vehicle limit

    % Rotate NED speed commands into the yaw-aligned body frame
    Vn_c = Vhorz_c * cos(chi_c);
    Ve_c = Vhorz_c * sin(chi_c);
    Vhorz_xyc = [cos(psi_hat)  sin(psi_hat);
                -sin(psi_hat)  cos(psi_hat)] * [Vn_c; Ve_c];
    Vhorz_xc = Vhorz_xyc(1);  % forward speed command in body frame, m/s
    Vhorz_yc = Vhorz_xyc(2);  % lateral speed command in body frame, m/s

    % Rotate NED speed estimates into the yaw-aligned body frame
    Vhorz_xy = [cos(psi_hat)  sin(psi_hat);
               -sin(psi_hat)  cos(psi_hat)] * [Vn_hat; Ve_hat];
    Vhorz_x = Vhorz_xy(1);  % current forward speed in body frame, m/s
    Vhorz_y = Vhorz_xy(2);  % current lateral speed in body frame, m/s

    % --- Controller gains ---
    % Gains are set here so they can be adjusted without re-running init_quadsim_params.
    % Each PIR controller uses (kp, ki, kd) tuned for the vehicle's response.

    % Roll: regulates roll angle via the aileron (differential roll) channel
    P.roll_kp = 0.01;
    P.roll_ki = 0.002;
    P.roll_kd = 0.006;

    % Altitude: regulates altitude via collective throttle
    P.altitude_kp = 0.07;
    P.altitude_ki = 0.015;
    P.altitude_kd = 0.08;

    % Pitch: regulates pitch angle via the elevator (forward tilt) channel
    P.pitch_kp = 0.012;
    P.pitch_ki = 0.002;
    P.pitch_kd = 0.01;

    % Yaw: regulates yaw angle via the rudder (differential yaw) channel
    P.yaw_kp = 0.1;
    P.yaw_ki = 0.012;
    P.yaw_kd = 0.1;

    % Forward velocity: regulates body-x speed by commanding pitch
    P.vhx_kp = 0.1;
    P.vhx_ki = 0.05;
    P.vhx_kd = 0.0;

    % Lateral velocity: regulates body-y speed by commanding roll
    P.vhy_kp = 0.1;
    P.vhy_ki = 0.05;
    P.vhy_kd = 0.0;

    % --- Autopilot execution ---
    persistent h_prev
    if firstTime
        h_prev = P.h0_m;
        % Reset all integrators at simulation start
        PIR_pitch_hold(0, 0, 0, firstTime, P);
        PIR_alt_hold_using_throttle(0, 0, 0, firstTime, P);
        PIR_roll_hold(0, 0, 0, firstTime, P);
        PIR_yaw_hold(0, 0, 0, firstTime, P);
    end

    % Altitude rate (finite difference over one time step)
    h_rate = -(h_prev - h_hat) / P.Ts;

    % Outer loop: velocity → attitude commands
    theta_c = -PIR_Vhorz_x_hold(Vhorz_xc, Vhorz_x, 0, firstTime, P);
    phi_c   =  PIR_Vhorz_y_hold(Vhorz_yc, Vhorz_y, 0, firstTime, P);

    % Inner loop: attitude → control deflections
    delta_e = PIR_pitch_hold(theta_c, theta_hat, q_hat, firstTime, P);
    delta_a = PIR_roll_hold(phi_c, phi_hat, p_hat, firstTime, P);
    delta_r = PIR_yaw_hold(psi_c, psi_hat, r_hat, firstTime, P);
    delta_t = PIR_alt_hold_using_throttle(h_c, h_hat, h_rate, firstTime, P);

    h_prev = h_hat;

    % Control channel output vector [elevator, aileron, rudder, throttle]
    delta = [delta_e; delta_a; delta_r; delta_t];

    % Manual override not supported for this vehicle model
    if P.manual_flight_flag
        error('Manual flight not supported in quadsim')
    end

    % Commanded state vector logged for plotting (see make_quadsim_plots.m)
    ap_command = [ ...
        Vhorz_c; ...  % commanded horizontal speed, m/s
        h_c;     ...  % commanded altitude, m
        chi_c;   ...  % commanded course, rad
        phi_c;   ...  % commanded roll, rad
        theta_c; ...  % commanded pitch, rad
        psi_c;   ...  % commanded yaw, rad
        0;       ...  % reserved
        0;       ...  % reserved
        0;       ...  % reserved
    ];

    % Concatenate controls and autopilot state: 4 + 9 = 13 outputs
    out = [delta; ap_command];

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR (PI with rate feedback) autopilot controllers
%
% Each controller regulates one scalar output using the form:
%
%   u = kp*(y_c - y) + ki*integral(y_c - y) - kd*y_dot
%
% Output saturation and integrator clamping prevent integrator wind-up
% when the output hits its limit.
%
% All controllers share the same signature:
%   u = PIR_xxx(y_c, y, y_dot, init_flag, P)
%
%   y_c       - setpoint (command)
%   y         - measured or estimated output
%   y_dot     - rate feedback (body rate proxy for the relevant axis)
%   init_flag - set true at t=0 to zero the persistent integrator
%   P         - parameter struct containing gains and saturation limits
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
% PIR_Vhorz_x_hold
%   Regulate body-x (forward) velocity by commanding a pitch angle.
%   Returns theta_c in radians, saturated at ±P.theta_max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_Vhorz_x_hold(Vhorz_xc, Vhorz_x_hat, q_hat, init_flag, P)

    kp = P.vhx_kp;
    ki = P.vhx_ki;
    kd = P.vhx_kd;
    u_lower_limit = -P.theta_max;
    u_upper_limit = +P.theta_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    error     = Vhorz_xc - Vhorz_x_hat;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_Vhorz_y_hold
%   Regulate body-y (lateral) velocity by commanding a roll angle.
%   Returns phi_c in radians, saturated at ±P.phi_max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_Vhorz_y_hold(Vhorz_yc, Vhorz_y_hat, q_hat, init_flag, P)

    kp = P.vhy_kp;
    ki = P.vhy_ki;
    kd = P.vhy_kd;
    u_lower_limit = -P.phi_max;
    u_upper_limit = +P.phi_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    error     = Vhorz_yc - Vhorz_y_hat;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_pitch_hold
%   Regulate pitch angle via the elevator (forward tilt) channel.
%   Returns a normalized elevator deflection, saturated at ±P.delta_e_max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_pitch_hold(theta_c, theta_hat, q_hat, init_flag, P)

    kp = P.pitch_kp;
    ki = P.pitch_ki;
    kd = P.pitch_kd;
    u_lower_limit = -P.delta_e_max;
    u_upper_limit = +P.delta_e_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    error     = theta_c - theta_hat;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int - kd*q_hat;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_alt_hold_using_throttle
%   Regulate altitude via collective throttle.
%   Altitude error is clamped to ±10 m to limit aggressive initial climbs.
%   Returns a throttle command, saturated at [P.delta_t_min, P.delta_t_max].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_alt_hold_using_throttle(h_c, h_hat, h_rate, init_flag, P)

    kp = P.altitude_kp;
    ki = P.altitude_ki;
    kd = P.altitude_kd;
    u_lower_limit = P.delta_t_min;
    u_upper_limit = P.delta_t_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    % Clamp altitude error to ±10 m to prevent aggressive initial response
    error = max(min(h_c - h_hat, 10), -10);

    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int - kd*h_rate;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_alt_hold_using_pitch
%   Alternative altitude controller that commands pitch angle instead
%   of throttle. Not used in the default autopilot configuration but
%   provided for reference and experimentation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_alt_hold_using_pitch(h_c, h_hat, q_hat, init_flag, P)

    kp = P.altitude_kp;
    ki = P.altitude_ki;
    kd = P.altitude_kd;
    u_lower_limit = -P.theta_max;
    u_upper_limit = +P.theta_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    error     = h_c - h_hat;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int - kd*q_hat;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_roll_hold
%   Regulate roll angle via the aileron (differential roll) channel.
%   Returns a normalized aileron deflection, saturated at ±P.delta_a_max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_roll_hold(phi_c, phi_hat, p_hat, firstTime, P)

    kp = P.roll_kp;
    ki = P.roll_ki;
    kd = P.roll_kd;
    u_lower_limit = -P.delta_a_max;
    u_upper_limit = +P.delta_a_max;

    persistent error_int;
    if firstTime
        error_int = 0;
    end

    error     = phi_c - phi_hat;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int - kd*p_hat;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIR_yaw_hold
%   Regulate yaw angle via the rudder (differential yaw) channel.
%   Angle error is wrapped to (-pi, pi] to avoid sign discontinuities
%   when crossing the ±180 degree boundary.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = PIR_yaw_hold(psi_c, psi_hat, r_hat, init_flag, P)

    kp = P.yaw_kp;
    ki = P.yaw_ki;
    kd = P.yaw_kd;
    u_lower_limit = -P.delta_r_max;
    u_upper_limit = +P.delta_r_max;

    persistent error_int;
    if init_flag
        error_int = 0;
    end

    % Wrap angular error to (-pi, pi] to avoid discontinuity at ±180 deg
    error     = mod(psi_c - psi_hat + pi, 2*pi) - pi;
    error_int = error_int + P.Ts * error;
    u = kp*error + ki*error_int - kd*r_hat;

    if u > u_upper_limit
        u = u_upper_limit;
        if ki*error > 0, error_int = error_int - P.Ts*error; end
    elseif u < u_lower_limit
        u = u_lower_limit;
        if ki*error < 0, error_int = error_int - P.Ts*error; end
    end
end
