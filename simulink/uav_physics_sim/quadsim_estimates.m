% quadsim_estimates.m
%
% Generation of feedback state estimates for quadsim
%
% Inputs:
%   Measurements
%   Time
%
% Outputs:
%   Feedback state estimates
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   
function out = quadsim_estimates(uu,P)

    % Extract variables from input vector uu
    %   uu = [meas(1:18); time(1)];
    k=(1:18);               meas=uu(k);   % Sensor Measurements
    k=k(end)+(1);           time=uu(k);   % Simulation time, s

    % Extract mesurements
    k=1;
    pn_gps = meas(k); k=k+1; % GPS North Measurement, m
    pe_gps = meas(k); k=k+1; % GPS East Measurement, m
    alt_gps= meas(k); k=k+1; % GPS Altitude Measurement, m
    Vn_gps = meas(k); k=k+1; % GPS North Speed Measurement, m/s
    Ve_gps = meas(k); k=k+1; % GPS East Speed Measurement, m/s
    Vd_gps = meas(k); k=k+1; % GPS Downward Speed Measurement, m/s
    p_gyro = meas(k); k=k+1; % Gyro Body Rate Meas. about x, rad/s
    q_gyro = meas(k); k=k+1; % Gyro Body Rate Meas. about y, rad/s
    r_gyro = meas(k); k=k+1; % Gyro Body Rate Meas. about z, rad/s
    ax_accel = meas(k); k=k+1; % Accelerometer Meas along x, m/s/s
    ay_accel = meas(k); k=k+1; % Accelerometer Meas along y, m/s/s
    az_accel = meas(k); k=k+1; % Accelerometer Meas along z, m/s/s
    static_press = meas(k); k=k+1; % Static Pressure Meas., N/m^2
    diff_press = meas(k); k=k+1; % Differential Pressure Meas., N/m^2
    psi_mag = meas(k); k=k+1; % Yaw Meas. from Magnetometer, rad
    olfa = meas(k); k=k+1;
    olfb = meas(k); k=k+1;
    olfc = meas(k); k=k+1;

    % Filter raw measurements
    persistent lpf_static_press ...
               lpf_diff_press ...
               lpf_p_gyro ...
               lpf_q_gyro ...
               lpf_r_gyro ...
               lpf_psi_mag
    if(time==0)
        % Filter initializations
        lpf_static_press = static_press;
        lpf_diff_press = diff_press;
        lpf_p_gyro = p_gyro;
        lpf_q_gyro = q_gyro;
        lpf_r_gyro = r_gyro;
        lpf_psi_mag = psi_mag;
    end
    lpf_static_press = LPF(static_press,lpf_static_press,P.tau_static_press,P.Ts);
    lpf_diff_press   = LPF(diff_press,lpf_diff_press,P.tau_diff_press,P.Ts);
    lpf_p_gyro = LPF(p_gyro,lpf_p_gyro,P.tau_gyro,P.Ts);
    lpf_q_gyro = LPF(q_gyro,lpf_q_gyro,P.tau_gyro,P.Ts);
    lpf_r_gyro = LPF(r_gyro,lpf_r_gyro,P.tau_gyro,P.Ts);
    lpf_psi_mag = LPF(psi_mag,lpf_psi_mag,P.tau_mag,P.Ts);
    
    % Estimate barometric altitude from static pressure
    P0 = 101325;  % Standard pressure at sea level, N/m^2
    R = 8.31432;  % Universal gas constant for air, N-m/(mol-K)
    M = 0.0289644;% Standard molar mass of atmospheric air, kg/mol
    T = 5/9*(P.air_temp_F-32)+273.15; % Air temperature in Kelvin
    P_launch = P0 * exp(-M * P.gravity / R / T * P.h0_ASL);
    h_baro = -R * T / M / P.gravity * log(lpf_static_press / P_launch); % Altitude estimate using Baro altimeter, meters above h0_ASL

    %% Note: zero out pressure since there is no pitot tube on quad-rotor
    % Va_pitot = sqrt(2 * lpf_diff_press / P.rho); % Airspeed estimate using pitot tube, m/s
    Va_pitot = 0.0;

    % EKF to estimate roll and pitch attitude
    sigma_trueMeasNoise = 5; % mx1 (units of measurements)
    sigma_ekfInitUncertainty = [5*pi/180 5*pi/180]; % nx1 (units of states)
    Q_att = diag([P.sigma_noise_gyro P.sigma_noise_gyro].^2);
    % The multiplication factor here should be around 1e4-1e5
    R_att = 1e4*diag([P.sigma_noise_accel P.sigma_noise_accel P.sigma_noise_accel].^2);
    persistent xhat_att P_att
    if(time==0)
        xhat_att=[0;0]; % States: [phi; theta]
        P_att=diag(sigma_ekfInitUncertainty.^2);
    end
    N=10; % Number of sub-steps for propagation each sample period
    dt = P.Ts;
    for i=1:N % Prediction step (N sub-steps)
        phi = xhat_att(1);
        theta = xhat_att(2);
        f_att_00 = p_gyro + (q_gyro*sin(phi)*tan(theta)) + (r_gyro*cos(phi)*tan(theta));
        f_att_10 = (q_gyro*cos(phi)) - (r_gyro*sin(phi));
        f_att = [f_att_00; f_att_10];
        A_att_00 = q_gyro*cos(phi)*tan(theta)-r_gyro*sin(phi)*tan(theta);
        A_att_01 = (q_gyro*sin(phi)+r_gyro*cos(phi))*(1+(tan(theta))^2);
        A_att_10 = -q_gyro*sin(phi)-r_gyro*cos(phi);
        A_att_11 = 0;
        A_att = [
            A_att_00 A_att_01;
            A_att_10 A_att_11
            ]; % Linearization (Jacobian) of f(x,...) wrt x
        xhat_att = xhat_att + (dt / N) * f_att; % States propagated to sub-step N
        P_att = P_att + ((dt / N) * ((A_att*P_att) + (P_att*A_att') + Q_att)); % Covariance matrix propagated to sub-step N
        P_att = real(.5*P_att + .5*P_att'); % Make sure P stays real and symmetric
    end
    Va = Va_pitot;
    phi = xhat_att(1);
    theta = xhat_att(2);
    y_att = [ax_accel; ay_accel; az_accel];
    h_att_00 = (q_gyro*Va*sin(theta)) + (P.gravity*sin(theta));
    h_att_10 = (r_gyro*Va*cos(theta)) - (p_gyro*Va*sin(theta)) - (P.gravity*cos(theta)*sin(phi));
    h_att_20 = (-q_gyro*Va*cos(theta)) - (P.gravity*cos(theta)*cos(phi));
    h_att = [h_att_00; h_att_10; h_att_20];
    C_att_00 = 0;
    C_att_01 = (q_gyro*Va*cos(theta)) + (P.gravity*cos(theta));
    C_att_10 = -P.gravity*cos(phi)*cos(theta);
    C_att_11 = (-r_gyro*Va*sin(theta)) - (p_gyro*Va*cos(theta)) + (P.gravity*sin(phi)*sin(theta));
    C_att_20 = P.gravity*sin(phi)*cos(theta);
    C_att_21 = (q_gyro*Va*sin(theta)) + (P.gravity*cos(phi)*sin(theta));
    C_att = [
        C_att_00 C_att_01;
        C_att_10 C_att_11;
        C_att_20 C_att_21
        ]; % Linearization (Jacobian) of h(x,...) wrt x
    L_att = (P_att*C_att')/((C_att*P_att*C_att') + R_att); % Kalman Gain matrix

    P_resid = C_att*P_att*C_att'+R_att; % Covariance matrix representing the combined predicted uncertainty of our estimate and measurement
    resid_x_unc = sqrt(P_resid(1,1)); % EKF predicted uncertainty of our estimate and measurement, mps2
    resid_y_unc = sqrt(P_resid(2,2)); % EKF predicted uncertainty of our estimate and measurement, mps2
    resid_z_unc = sqrt(P_resid(3,3)); % EKF predicted uncertainty of our estimate and measurement, mps2
    resid = y_att-h_att; % EKF Measurement residual

    I = eye(length(xhat_att)); % nxn identity matrix
    P_att = (I - (L_att*C_att))*P_att; % Covariance matrix updated with measurement information
    xhat_att = xhat_att + (L_att*(y_att - h_att)); % States updated with measurement information
    xhat_att = mod(xhat_att+pi,2*pi)-pi; % xhat_att are attitudes, make sure they stay within +/-180degrees 
    phi_hat_unc   = sqrt(P_att(1,1)); % EKF-predicted uncertainty in phi estimate, rad 
    theta_hat_unc = sqrt(P_att(2,2)); % EKF-predicted uncertainty in theta estimate, rad 


    %% GPS EKF
    % Use GPS for NE position, and ground velocity vector
    R_ned2b = eulerToRotationMatrix(xhat_att(1), xhat_att(2), lpf_psi_mag);
    eta_pn = (P.sigma_eta_gps_north);
    eta_pe = (P.sigma_eta_gps_east);
    eta_alt = (P.sigma_eta_gps_alt);
    eta_Vn = P.sigma_noise_gps_speed;
    eta_Ve = P.sigma_noise_gps_speed;
    eta_Vd = P.sigma_noise_gps_speed;
    sigma_ekfInitUncertainty_gps = [eta_pn eta_pe eta_alt eta_Vn eta_Ve eta_Vd]; % nx1 (units of states)
    Q_gps = diag([.5; .5; .5; .3; .3; .3].^2);
    R_gps = diag([2 2 2 .1 .1 .1].^2);
    persistent xhat_gps P_gps
    if(time==0)
        xhat_gps = [pn_gps;pe_gps;-alt_gps;Vn_gps;Ve_gps;Vd_gps];
%         P_gps=diag(sigma_ekfInitUncertainty_gps);
        P_gps=diag([5; 5; 5; 2; 2; 2].^2);
    end
    dt_gps=P.Ts;
    N=10; % Number of sub-steps for propagation each sample period
    for i=1:N % Prediction step (N sub-steps)
        accel_gps = R_ned2b' * [ax_accel; ay_accel; az_accel];
        f_gps = [
            xhat_gps(4);
            xhat_gps(5);
            xhat_gps(6);
            accel_gps(1);
            accel_gps(2);
            accel_gps(3) + P.gravity
            ];
        A_gps = [
            0 0 0 1 0 0;
            0 0 0 0 1 0;
            0 0 0 0 0 1;
            zeros(3, 6); % Since f_gps(4:6) are not a function of xhat, A(4:6,:) = zeros(3,6)
            ]; % Linearization (Jacobian) of f(x,...) wrt x
        
        xhat_gps = xhat_gps + (dt_gps / N) * f_gps; % States propagated to sub-step N
        P_gps = P_gps + ((dt_gps / N) * ((A_gps*P_gps) + (P_gps*A_gps') + Q_gps)); % Covariance matrix propagated to sub-step N
        P_gps = real(.5*P_gps + .5*P_gps'); % Make sure P stays real and symmetric
    end
    persistent prev_pn_gps prev_pe_gps
    if time==0
        prev_pn_gps=0;
        prev_pe_gps=0;
    end
    if (pn_gps~=prev_pn_gps) || (pe_gps~=prev_pe_gps)
        prev_pn_gps=pn_gps;
        prev_pe_gps=pe_gps;
        y_gps = [pn_gps; pe_gps; -alt_gps; Vn_gps; Ve_gps; Vd_gps];
        h_gps = xhat_gps;
        C_gps = eye(6);% Linearization (Jacobian) of h(x,...) wrt x
        %L_gps = (P_gps*C_gps')*inv((C_gps*P_gps*C_gps') + R_gps); % Kalman Gain matrix
        L_gps = (P_gps*C_gps')/((C_gps*P_gps*C_gps') + R_gps); % Kalman Gain matrix
        I = eye(size(P_gps)); % nxn identity matrix
        P_gps = (I - (L_gps*C_gps))*P_gps; % Covariance matrix updated with measurement information
        P_gps = real(.5*P_gps + .5*P_gps'); % Make sure P stays real and symmetric
        xhat_gps = xhat_gps + (L_gps*(y_gps - h_gps)); % States updated with measurement information
    end

    %% Note: not used but could prove helpful later
    pn_hat_unc = sqrt(P_gps(1,1));
    pe_hat_unc = sqrt(P_gps(2,2));
    h_hat_unc = sqrt(P_gps(3,3));
    Vn_hat_unc = sqrt(P_gps(4,4));
    Ve_hat_unc = sqrt(P_gps(5,5));
    Vd_hat_unc = sqrt(P_gps(6,6));

    % estimate states
    pn_hat    = xhat_gps(1);
    pe_hat    = xhat_gps(2);
    h_hat     = h_baro;
    Va_hat    = Va_pitot;
    phi_hat   = xhat_att(1);
    theta_hat = xhat_att(2);
    psi_hat   = lpf_psi_mag;
    p_hat     = lpf_p_gyro;
    q_hat     = lpf_q_gyro;
    r_hat     = lpf_r_gyro;
    Vn_hat    = xhat_gps(4);
    Ve_hat    = xhat_gps(5);
    Vd_hat    = xhat_gps(6);
    wn_hat    = 0;
    we_hat    = 0;
    
    % Compile output vector
    out = [...
            pn_hat;...    % 1
            pe_hat;...    % 2
            h_hat;...     % 3
            Va_hat;...    % 4
            phi_hat;...   % 5
            theta_hat;... % 6
            psi_hat;...   % 7
            p_hat;...     % 8
            q_hat;...     % 9 
            r_hat;...     % 10
            Vn_hat;...    % 11
            Ve_hat;...    % 12
            Vd_hat;...    % 13
            wn_hat;...    % 14
            we_hat;...    % 15
            phi_hat_unc;...   % 16
            theta_hat_unc;... % 17
            resid(1); % 18
            resid(2); % 19
            resid(3); % 20
            resid_x_unc; % 21
            resid_y_unc; % 22
            resid_z_unc; % 23
        ]; % Length: 23
    
end 


function y = LPF(u,yPrev,tau,Ts)
%
%  Y(s)       a           1
% ------ = ------- = -----------,  tau: Filter time contsant, s
%  U(s)     s + a     tau*s + 1         ( tau = 1/a )
%

    % s = tf('s');
    % a = 1/tau;
    alpha_LPF = exp(-Ts / tau);
    y = (alpha_LPF * yPrev) + ((1 - alpha_LPF) * u);
end
