% Quadcopter physical properties
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012


% physical parameters of airframe
P.mass = 1.023;   % kg
P.Jx   = 0.0095;  % kg-m^2
P.Jy   = 0.0095;  % kg-m^2
P.Jz   = 0.0186;  % kg-m^2
P.Jxz  = 0;       % kg-m^2
P.L_to_motor = 0.2223; % Arm length from hub to motors (at 45deg in X configuration)

% Prop params
P.S_prop        = pi*.111^2;          % Aero swept by prop, m^2
P.C_prop        = 1;                  % Prop efficientcy coefficient, no units
P.k_motor       = 17.4556;            % Motor constant, m/s
P.k_Tp          = 2.6673e-7;          % Prop torque constant, kg-m^2 = N-m/(rad/s)^2
P.k_omega       = 946.1011;           % Prop speed constant, rad/s (Equiv. to 9034.6 RPM)

% lumped parameters
P.mu_rotorDrag = 0.52;                % Total rotor drag coefficient, N/(m/s)

% Propeller Rotation Speed Miscalibration Biases
%   ALWAYS keep these values zeroed in this file.
%   You can manually set biases from the workspace
%   AFTER you trim and linearize.
%   You do NOT want biases corrupting the autopilot tuning process.
P.prop_1_omega_bias = 0;  % rad/s
P.prop_2_omega_bias = 0;  % rad/s
P.prop_3_omega_bias = 0;  % rad/s
P.prop_4_omega_bias = 0;  % rad/s

