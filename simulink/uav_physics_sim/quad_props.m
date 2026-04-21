% Quadcopter physical properties — DJI Tello
%
% Sets vehicle-specific physical parameters on the P struct.
% Called by init_quadsim_params.m during simulation initialization.
% Edit values here to model a different quadrotor airframe.
%
% DJI Tello specifications (source: DJI/RyzeTech product documentation):
%   Mass (no guards):   ~80 g
%   Dimensions:         98 × 92.5 × 41 mm (with guards)
%   Max horizontal speed: ~8 m/s
%   Max flight time:    ~13 min
%
% NOTE: Moments of inertia and motor parameters (k_motor, k_Tp, k_omega)
% are estimated from physical dimensions and hover analysis. For precise
% control, verify these through system identification on the actual vehicle.
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012

% --- Airframe ---
P.mass = 0.080;    % kg — bare vehicle without propeller guards
%                       Add ~0.007 kg for guards, more for OPU payload.
P.Jx   = 4.5e-5;   % kg-m^2 — roll axis (estimated from frame geometry)
P.Jy   = 4.5e-5;   % kg-m^2 — pitch axis (estimated; Tello is nearly square)
P.Jz   = 8.5e-5;   % kg-m^2 — yaw axis (estimated; sum rule from Jx, Jy + motor mass)
P.Jxz  = 0;        % kg-m^2 — zero for symmetric X-frame
P.L_to_motor = 0.050; % m — arm length from body center to motor (at 45 deg in X-frame)
%                         Estimated from 98×92.5 mm outer dimensions.

% --- Propeller ---
% Tello props are approximately 60 mm diameter (radius = 0.030 m).
P.S_prop   = pi * 0.030^2;  % 0.002827 m^2 — aero area swept per propeller
P.C_prop   = 1.0;            % prop efficiency coefficient (dimensionless)

% k_motor is tuned so that the hover throttle (delta_t0) comes out near 0.5.
%   delta_t0 = sqrt( m*g / (4 * rho * C_prop * S_prop * k_motor^2) )
%   Solving for k_motor at delta_t0 = 0.5:
%   k_motor = sqrt( m*g / (4 * rho * C_prop * S_prop * 0.25) ) ≈ 14.8 m/s
P.k_motor  = 14.8;           % m/s — motor speed constant (Beard & McClain momentum model)

% Prop torque constant: torque = k_Tp * omega_prop^2
%   Estimated from hover: T_hover/motor ≈ 0.196 N, Q/T ratio ≈ 0.014 m,
%   omega_hover ≈ 1050 rad/s → k_Tp = Q / omega^2 ≈ 2.7e-9 N·m·s^2/rad^2
P.k_Tp     = 2.7e-9;         % N-m/(rad/s)^2 — prop torque constant

% Tello brushed motors reach approximately 20,000 RPM at full throttle.
%   20,000 RPM × (2*pi / 60) = 2094 rad/s
P.k_omega  = 2094;           % rad/s — max prop angular speed (~20,000 RPM)

% --- Drag ---
% Lumped rotor drag: scaled from a larger vehicle by mass and prop area ratio.
P.mu_rotorDrag = 0.04;       % N/(m/s) — total translational drag from rotors

% --- Propeller speed miscalibration biases ---
%   ALWAYS keep these zeroed in this file.
%   You can manually set non-zero biases from the workspace AFTER trim and
%   linearization. Setting them here would corrupt the autopilot tuning.
P.prop_1_omega_bias = 0;  % rad/s
P.prop_2_omega_bias = 0;  % rad/s
P.prop_3_omega_bias = 0;  % rad/s
P.prop_4_omega_bias = 0;  % rad/s
