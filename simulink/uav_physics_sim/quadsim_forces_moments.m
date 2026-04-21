% quadsim_forces_moments.m
%
% Generation of forces and moments acting on vehicle for quadsim
%
% Inputs:
%   Wind in NED frame
%   Control surfaces
%   UAV States
%   Time
%
% Outputs:
%   Forces in body frame
%   Moments in body frame
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   
function out = quadsim_forces_moments(uu, P)

    % Extract variables from input vector uu
    %   uu = [wind_ned(1:3); deltas(1:4); x(1:12); time(1)];
    k=(1:3);          wind_ned=uu(k);   % Total wind vector, ned, m/s
    k=k(end)+(1:4);   deltas=uu(k);     % Control surface commands: [delta_e delta_a delta_r delta_t]
    k=k(end)+(1:12);  x=uu(k);          % states
    k=k(end)+(1);     time=uu(k);       % Simulation time, s

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

    % Combine states to vector form for convenience
    P_ned = [pn; pe; pd];   % NED position, m
    vg_b  = [u; v; w];      % Groundspeed vector, body frame, m/s
    w_b   = [p; q; r];      % body rates about x,y,z, rad/s

    % Extract control commands from deltas
    delta_e = deltas(1); % Elevator, +/-
    delta_a = deltas(2); % Aileron, +/-
    delta_r = deltas(3); % Rudder, +/-
    delta_t = deltas(4); % Throttle, 0 - 1
    [delta_1, delta_2, delta_3, delta_4] = mapChannelsToMotors(delta_e,delta_a,delta_r,delta_t);

    % Prop rotation rates
    omega_1 = P.k_omega * delta_1 + P.prop_1_omega_bias;  % Propeller 1 rotation rate, rad/s (function of P.prop_1_omega_bias, etc.)
    omega_2 = P.k_omega * delta_2 + P.prop_2_omega_bias;  % Propeller 2 rotation rate, rad/s (function of P.prop_1_omega_bias, etc.)
    omega_3 = P.k_omega * delta_3 + P.prop_3_omega_bias;  % Propeller 3 rotation rate, rad/s (function of P.prop_1_omega_bias, etc.)
    omega_4 = P.k_omega * delta_4 + P.prop_4_omega_bias;  % Propeller 4 rotation rate, rad/s (function of P.prop_1_omega_bias, etc.)

    R_ned2b = eulerToRotationMatrix(phi, theta, psi);
    % compute wind vector in body frame (wind_ned is an input)
    wind_b = R_ned2b * wind_ned;

    % compute airspeed Va, angle-of-attack alpha, side-slip beta
    va_rel_b=vg_b-wind_b;
    [Va alpha beta] = makeVaAlphaBeta(va_rel_b);

    % Create and combine Forces 
    %% gravity
    f_grav_ned =     P.mass*[0; 0; P.gravity];
    f_grav_b = R_ned2b * f_grav_ned;

    %% aero forces
    f_dyn_press = 0.5*P.rho*Va*Va;
    
    % aero drag force
    f_rotor_drag = -P.mu_rotorDrag * [u - wind_b(1); v - wind_b(2); 0];
    f_body_drag = 0; % approximately
    f_drag = f_rotor_drag + f_body_drag;

    %% aero force props (lift)
    % for a fixed wing aircraft, V_air_in ~=V_a
    V_air_in = -va_rel_b(3); % negative of 3rd compeonent of wind-rel velocity vector in body
    f_prop_1 = P.rho*P.C_prop*P.S_prop*(V_air_in + (omega_1/P.k_omega)*(P.k_motor-V_air_in))*((omega_1/P.k_omega)*(P.k_motor-V_air_in));
    f_prop_2 = P.rho*P.C_prop*P.S_prop*(V_air_in + (omega_2/P.k_omega)*(P.k_motor-V_air_in))*((omega_2/P.k_omega)*(P.k_motor-V_air_in));
    f_prop_3 = P.rho*P.C_prop*P.S_prop*(V_air_in + (omega_3/P.k_omega)*(P.k_motor-V_air_in))*((omega_3/P.k_omega)*(P.k_motor-V_air_in));
    f_prop_4 = P.rho*P.C_prop*P.S_prop*(V_air_in + (omega_4/P.k_omega)*(P.k_motor-V_air_in))*((omega_4/P.k_omega)*(P.k_motor-V_air_in));
    f_prop_b = -(f_prop_1+f_prop_2+f_prop_3+f_prop_4) * [0; 0; 1];
    f_b = f_grav_b + f_drag + f_prop_b;

    torque_prop_1 = P.k_Tp*omega_1*omega_1;
    torque_prop_2 = P.k_Tp*omega_2*omega_2;
    torque_prop_3 = P.k_Tp*omega_3*omega_3;
    torque_prop_4 = P.k_Tp*omega_4*omega_4;
    delta_y = P.L_to_motor*cos(45*pi/180);
    delta_x = P.L_to_motor*sin(45*pi/180);
    m_x = (delta_y*(f_prop_3 + f_prop_2)- delta_y*(f_prop_1 + f_prop_4)) * [1; 0; 0];
    m_y = (delta_x*(f_prop_1 + f_prop_3)- delta_x*(f_prop_2 + f_prop_4)) * [0; 1; 0];
    m_z = ((torque_prop_1 + torque_prop_2) - (torque_prop_3 + torque_prop_4)) * [0; 0; 1];

    % Create and combine Moments    
    m_b = m_x + m_y + m_z;

    % Compile function output
    out = [f_b; m_b]; % Length 3+3=6
    
end
