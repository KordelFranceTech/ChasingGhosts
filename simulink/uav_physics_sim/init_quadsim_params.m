% init_quadsim_params.m
%
% Initialize parameters structure (P) for quadsim
%
% Inputs:
%   None
%
% Outputs:
%   P: parameters structure used for quadsim
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   
function P = init_quadsim_params

    P = [];

    % Acquire UAV vehicle model params
    %   e.g. P.mass_kg, P.C_L_alpha, etc.
    quad_props

    P.Ts = 0.01;     % Autopilot sample time step, s

    P.Tlog = 0.01;   % Logging time step, s
    P.Tvis = 0.2;    % Visualization time step, s
    
    P.manual_flight_flag=0; % Perform manually controlled flight
    
    % environment params
    P.gravity = 9.80665; % m/s^2
    P.rho = 1.2682;      % air density, kg/m^3
    P.h0_ASL = 123;      % height above sea level of NED reference location, m (used in static pressure measurement)
    P.air_temp_F = 72;   % air temperature, Fahrenheit (used in static pressure measurement)

    % wind parameters
    P.wind_n = 0;        % North component of steady wind vector, m/s
    P.wind_e = 0;        % East component of steady wind vector, m/s
    P.wind_d = 0;        % Down component of steady wind vector, m/s
    P.L_wx = 200;        % Dryden gusting spatial wavelength along body x, m
    P.L_wy = 200;        % Dryden gusting spatial wavelength along body y, m
    P.L_wz = 50;         % Dryden gusting spatial wavelength along body z, m
    P.sigma_wx = 1.06;   % Dryden intensity along body x, m/s 
    P.sigma_wy = 1.06;   % Dryden intensity along body y, m/s
    P.sigma_wz = 0.7;    % Dryden intensity along body z, m/s
    P.Va_dryden = 10;    % Airspeed parameter for modified Dryden, m/s

    % control limit parameters
    P.delta_e_max =  0.1;  % Max allowable elevator channel command, -1 to 1
    P.delta_a_max =  0.1;  % Max allowable aileron channel command, -1 to 1
    P.delta_r_max =  0.1;  % Max allowable rudder channel command, -1 to 1
    P.delta_t_max =  0.9;  % Max allowable throttle channel command, -1 to 1
    P.delta_t_min =  0.1;  % Min allowable throttle channel command, -1 to 1
    P.theta_max   =  25*pi/180;  % Max allowable pitch command, rad
    P.phi_max     =  25*pi/180;  % Max allowable roll command, rad
    P.psi_max     =  pi; % Max allowable yaw command, rad

    % specify nominal launch height
    P.h0_m = 50;    % launch altitude, m
    
    % first cut at initial conditions
    P.Vhorz_max = 8;
    
    % Dynamic States
    P.pn0    = 0;  % initial North position, m
    P.pe0    = 0;  % initial East position, m
    P.pd0    = -P.h0_m;  % initial Down position (negative altitude), m
    P.u0     = 0;  % initial ground-relative velocity along body x-axis, m/s
    P.v0     = 0;  % initial  ground-relative velocity along body y-axis, m/s
    P.w0     = 0;  % initial  ground-relativevelocity along body z-axis, m/s
    P.phi0   = 0;  % initial roll angle, rad, (+rightWingDown)
    P.theta0 = 0;  % initial pitch angle, rad (+noseUp)
    P.psi0   = 0;  % initial yaw angle, rad (+EastOfNorth)
    P.p0     = 0;  % initial body frame roll rate, rad/s
    P.q0     = 0;  % initial body frame pitch rate, rad/s
    P.r0     = 0;  % initial body frame yaw rate, rad/s

    P.delta_e0 = 0;  % initial elevator channel, -1 to 1
    P.delta_a0 = 0;  % initial aileron channel, -1 to 1
    P.delta_r0 = 0;  % initial rudder channel, -1 to 1
    P.delta_t0 = 0.5;  % initial throttle value, 0-1

    % Note: A quad does not have a pitot tube for differential pressure
    % measurement. These parameters are retained for structural consistency
    % with fixed-wing sensor/estimates modules.

    % Sensor parameters: Gyro and Accelerometer
    P.sigma_noise_gyro = .015*sqrt(80)*pi/180; % rad/s
    P.sigma_noise_accel = 250*sqrt(100)/1e6*P.gravity; % m/s^2

    % Sensor parameters: Pressure Sensors
    kPa_to_Npm2 = 1000; % 1 Pa = 1 N/m^2, so 1 kPa = 1000 N/m^2
    P.sigma_bias_static_press = 0.04*kPa_to_Npm2; % Static pressure measurement bias std. dev., N/m^2
    P.sigma_noise_static_press = 0.01*kPa_to_Npm2; % Static pressure measurement noise std. dev., N/m^2
    P.sigma_bias_diff_press = 0.020*kPa_to_Npm2; % Differential pressure measurement bias std. dev., N/m^2
    P.sigma_noise_diff_press = 0.002*kPa_to_Npm2; % Differential pressure measurement noise std. dev., N/m^2

    % Sensor parameters: Magnetometer
    P.sigma_bias_mag = 1*pi/180; % Magnetometer bias std. dev., rad
    P.sigma_noise_mag = 0.3*pi/180; % Magnetometer noise std. dev., rad

    % Sensor parameters: GPS
    P.Ts_gps = 1; % GPS sampling time, s
    P.sigma_bias_gps_north = 4.7;  % GPS north Gauss-Markov bias parameter, m
    P.sigma_bias_gps_east  = 4.7;  % GPS east  Gauss-Markov bias parameter, m
    P.sigma_bias_gps_alt   = 9.2;  % GPS alt   Gauss-Markov bias parameter, m
    P.sigma_eta_gps_north = 0.21;  % GPS north Gauss-Markov noise parameter, m
    P.sigma_eta_gps_east  = 0.21;  % GPS east  Gauss-Markov noise parameter, m
    P.sigma_eta_gps_alt   = 0.40;  % GPS alt   Gauss-Markov noise parameter, m
    P.tau_gps = 1100;  % GPS Gauss-Markov time constant, s (In book: k_gps = 1/tau_gps)
    P.sigma_noise_gps_speed = 0.1; % GPS groundspeed std. dev., m/s

    % State estimation parameters
    P.tau_static_press = 0.5;      % Time constant for autopilot's static pressure LPF, seconds
    P.tau_diff_press = 0.25;       % Time constant for autopilot's differential pressure LPF, seconds
    P.tau_gyro = 0.05;             % Time constant for autopilot's gyros, seconds
    P.tau_accel = 0.05;            % Time constant for autopilot's accelerometers, seconds
    P.tau_mag = 0.2;               % Time constant for autopilot's magnetometer, seconds

    % Electrochemical detection parameters
    P.sigma_bias_echem = 1.0;
    P.sigma_noise_echem = 1.0;
