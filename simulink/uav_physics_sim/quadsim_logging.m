% quadsim_logging.m
%
% Logging of quadsim variables
%
% Inputs:
%   Various 
%
% Outputs:
%   Creates a "out" structure in the Matlab workspace
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012
function quadsim_logging(uu,P)

    % Logging flags
    log_commands     = 1;
    log_measurements = 1;
    log_estimates    = 1;
    
    % Extract variables from input vector uu
    %   uu = [x(1:12); f_and_m(1:6); wind_ned(1:3); ap_cmds(1:9); estimates(1:23); meas(1:18); time(1)];
    k=1:12;          x=uu(k);         % States
    k=k(end)+(1:6);  f_and_m=uu(k);   % Forces and Moments, body
    k=k(end)+(1:3);  wind_ned=uu(k);  % Wind vector, ned, m/s
    k=k(end)+(1:4);  deltas=uu(k);    % Control commands
    k=k(end)+(1:9);  ap_cmds=uu(k);   % Autopilot commands
    k=k(end)+(1:23); estimates=uu(k); % Autopilot state estimates
    k=k(end)+(1:18); meas=uu(k);      % Measurements
    k=k(end)+(1);    time=uu(k);      % Simulation time, s

    % Extract state variables from x
    pn    = x(1);   % North position, m
    pe    = x(2);   % East position, m
    pd    = x(3);   % Down position, m
    u     = x(4);   % body-x groundspeed component, m/s
    v     = x(5);   % body-y groundspeed component, m/s
    w     = x(6);   % body-z groundspeed component, m/s
    phi   = x(7);   % EulerAngle: roll, rad
    theta = x(8);   % EulerAngle: pitch, rad
    psi   = x(9);   % EulerAngle: yaw, rad
    p     = x(10);  % body rate about x, rad/s
    q     = x(11);  % body rate about y, rad/s
    r     = x(12);  % body rate about z, rad/s

    % Extract control commands
    delta_e   = deltas(1); % Elevator, rad
    delta_a   = deltas(2); % Aileron, rad
    delta_r   = deltas(3); % Rudder, rad
    delta_t   = deltas(4); % Throttle, 0-1

    % Compute Rotation Matrices
    R_ned2b = eulerToRotationMatrix(phi,theta,psi);
    R_b2ned = R_ned2b';

    % Manipulate states
    vg_ned = R_b2ned*[u; v; w];
    [Vg gamma course] = makeVgGammaCourse(vg_ned);

    % Extract body-frame forces and moments
    f_b   = f_and_m(1:3); % External forces along body x,y,z, N
    m_b   = f_and_m(4:6); % External moments about body x,y,z, N-m

    % Use a persistent variable for incrementing logging index
    % Re-initialize "out" structure if first time through
    persistent i
    if time==0
        i=0;
        out = [];
        assignin('base','out',out);
    end
    i=i+1;

    % Acquire out structure from base workspace
    out = evalin('base','out');

    % Append new data to out structure
    % (Note: This is highly inefficient, but it is convenient)

    out.time_s(i) = time;

    out.north_m(i) = pn;
    out.east_m(i) = pe;
    out.alt_m(i) = -pd;
    out.groundspeed_mps(i) = Vg;
    out.gamma_deg(i) = gamma*180/pi;    
    out.course_deg(i) = mod(course*180/pi+180,360)-180; % Limit: -180 to 180
    out.roll_deg(i) = mod(phi*180/pi+180,360)-180; % Limit: -180 to 180
    out.pitch_deg(i) = theta*180/pi;
    out.yaw_deg(i) = mod(psi*180/pi+180,360)-180; % Limit: -180 to 180
    out.p_dps(i) = p*180/pi;
    out.q_dps(i) = q*180/pi;
    out.r_dps(i) = r*180/pi;
    out.alt_rate_mps(i) = Vg*sin(gamma);
    out.horz_speed_mps(i) = Vg*cos(gamma);

    out.wind_north_mps(i) = wind_ned(1);
    out.wind_east_mps(i) = wind_ned(2);
    out.wind_down_mps(i) = wind_ned(3);

    out.delta_e(i) = delta_e;
    out.delta_a(i) = delta_a;
    out.delta_r(i) = delta_r;
    out.delta_t(i) = delta_t;
    [delta_1, delta_2, delta_3, delta_4] = mapChannelsToMotors(delta_e,delta_a,delta_r,delta_t);
    out.delta_1(i) = delta_1;
    out.delta_2(i) = delta_2;
    out.delta_3(i) = delta_3;
    out.delta_4(i) = delta_4;
    out.omega_1_rps(i) = (P.k_omega*delta_1 + P.prop_1_omega_bias); % radians/sec ( scale by   )
    out.omega_2_rps(i) = (P.k_omega*delta_2 + P.prop_2_omega_bias); % radians/sec (  2*pi/60   )
    out.omega_3_rps(i) = (P.k_omega*delta_3 + P.prop_3_omega_bias); % radians/sec ( to display )
    out.omega_4_rps(i) = (P.k_omega*delta_4 + P.prop_4_omega_bias); % radians/sec (    RPM     )

    out.fx_N(i) = f_b(1);
    out.fy_N(i) = f_b(2);
    out.fz_N(i) = f_b(3);
    out.mx_Nm(i) = m_b(1);
    out.my_Nm(i) = m_b(2);
    out.mz_Nm(i) = m_b(3);

    % Extract a/p commands
    if log_commands
        out.horz_speed_cmd_mps(i)= ap_cmds(1);
        out.alt_cmd_m(i)         = ap_cmds(2);
        out.course_cmd_deg(i)    = mod(ap_cmds(3)*180/pi+180,360)-180; % Limit: -180 to 180
        out.roll_cmd_deg(i)      = mod(ap_cmds(4)*180/pi+180,360)-180; % Limit: -180 to 180
        out.pitch_cmd_deg(i)     = ap_cmds(5)*180/pi;
        out.yaw_cmd_deg(i)       = mod(ap_cmds(6)*180/pi+180,360)-180; % Limit: -180 to 180
    end
    
    % Extract sensor measurements
    if log_measurements
        out.north_gps_m(i)     = meas(1); 
        out.east_gps_m(i)      = meas(2);
        out.alt_gps_m(i)       = meas(3);
        [Vg_gps gamma_gps course_gps] = makeVgGammaCourse(meas(4:6));
        out.groundspeed_gps_mps(i) = Vg_gps;
        out.gamma_gps_deg(i)   = 180/pi*gamma_gps;
        out.course_gps_deg(i)  = 180/pi*course_gps;
        out.alt_rate_gps_mps(i)   = Vg_gps*sin(gamma_gps);
        out.horz_speed_gps_mps(i) = Vg_gps*cos(gamma_gps);
        out.p_gyro_dps(i)      = meas(7)*180/pi;
        out.q_gyro_dps(i)      = meas(8)*180/pi;
        out.r_gyro_dps(i)      = meas(9)*180/pi;
        out.ax_accel_mps2(i)   = meas(10);
        out.ay_accel_mps2(i)   = meas(11);
        out.az_accel_mps2(i)   = meas(12);
        out.static_press_Npm2(i) = meas(13);
        out.diff_press_Npm2(i)   = meas(14);
        out.yaw_mag_deg(i)     = mod(meas(15)*180/pi+180,360)-180; % Limit: -180 to 180
    end

    % Extract state estimates
    if log_estimates
        out.north_est_m(i)         = estimates(1);
        out.east_est_m(i)         = estimates(2);
        out.alt_baro_m(i)         = estimates(3);
        out.airspeed_pitot_mps(i) = estimates(4);
        out.roll_est_deg(i) = mod(estimates(5)*180/pi+180,360)-180; % Limit: -180 to 180
        out.pitch_est_deg(i) = estimates(6)*180/pi;
        out.yaw_est_deg(i) = mod(estimates(7)*180/pi+180,360)-180; % Limit: -180 to 180
        out.p_est_dps(i) = estimates(8)*180/pi;
        out.q_est_dps(i) = estimates(9)*180/pi;
        out.r_est_dps(i) = estimates(10)*180/pi; 
        [Vg_est gamma_est course_est] = makeVgGammaCourse(estimates(11:13));
        out.groundspeed_est_mps(i) = Vg_est;
        out.gamma_est_deg(i) = gamma_est*180/pi;
        out.course_est_deg(i) = course_est*180/pi;
        out.alt_rate_est_mps(i) = Vg_est*sin(gamma_est);
        out.horz_speed_est_mps(i) = Vg_est*cos(gamma_est);
        out.phi_hat_unc(i) = estimates(16)*180/pi;
        out.theta_hat_unc(i) = estimates(17)*180/pi;
    end
    
    % For profiling purposes
    persistent tNow0
    if time==0 tNow0=now*24*3600; end
    out.real_time_s(i) = now*24*3600-tNow0; % elapsed wall clock time, s
    
    % Write out structure back to workspace
    assignin('base','out',out);

    % Apply stopping conditions
    %  - Ground
    %  - Extreme accelerations or body rates (linear aero model isn't valid
    %    at extreme conditions):
    %      - 400 deg/s
    if (-pd < 0) && (gamma<0)
        fprintf('Sim stopped due to ground clobber\n');
        error(['Sim stopped: Altitude below ground (see ' mfilename '.m)'])
    end
    if abs(p)*180/pi>400 || abs(q)*180/pi>400 || abs(r)*180/pi>400
        fprintf('Sim stopped due to excessive body rates:\n');
        fprintf('    p = %.1f deg\n',p*180/pi)
        fprintf('    q = %.1f deg\n',q*180/pi)
        fprintf('    r = %.1f deg\n',r*180/pi)
        error(['Sim stopped: Excessive body rate (see ' mfilename '.m)'])
    end
    
end
