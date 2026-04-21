% quadsim_kin_dyn.m
%
% Rigid-body kinematics and dynamics for the quadrotor.
%
% Integrates the 12 nonlinear equations of motion to produce state
% derivatives given the current state and applied forces/moments.
%
% State vector (12 elements):
%   [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
%   Positions (NED, m), body velocities (m/s), Euler angles (rad),
%   and body angular rates (rad/s).
%
% Inputs:
%   uu = [x(1:12); f_and_m(1:6); time(1)]
%
% Outputs:
%   out = xdot (12x1 state derivative vector)
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012
function out = quadsim_kin_dyn(uu, P)

    % Unpack input vector
    %   uu = [x(1:12); f_and_m(1:6); time(1)]
    k=1:12;          x=uu(k);        % 12-state vector
    k=k(end)+(1:6);  f_and_m=uu(k);  % Forces and moments, body frame
    k=k(end)+(1);    time=uu(k);     % Simulation time, s

    % Extract state variables
    pn    = x(1);   % North position, m
    pe    = x(2);   % East position, m
    pd    = x(3);   % Down position, m (negative = above ground)
    u     = x(4);   % Body-x groundspeed, m/s
    v     = x(5);   % Body-y groundspeed, m/s
    w     = x(6);   % Body-z groundspeed, m/s
    phi   = x(7);   % Roll angle, rad
    theta = x(8);   % Pitch angle, rad
    psi   = x(9);   % Yaw angle, rad
    p     = x(10);  % Body roll rate, rad/s
    q     = x(11);  % Body pitch rate, rad/s
    r     = x(12);  % Body yaw rate, rad/s

    vg_b = [u; v; w];   % Groundspeed vector in body frame
    w_b  = [p; q; r];   % Angular rate vector in body frame

    % External forces and moments in body frame
    f_b = f_and_m(1:3);  % Net force along body x, y, z, N
    m_b = f_and_m(4:6);  % Net moment about body x, y, z, N-m

    % Rotation matrix from NED to body frame (ZYX Euler angles)
    R_ned2b = eulerToRotationMatrix(phi, theta, psi);

    % --- NED position kinematics ---
    % Pdot = R' * vg_b  (rotate body velocity into NED frame)
    Pdot_ned = R_ned2b' * vg_b;

    % --- Body velocity dynamics (Newton's 2nd law in rotating frame) ---
    % vgdot = -omega x vg + F/m
    vgdot_b = -cross(w_b, vg_b) + (1/P.mass) * f_b;

    % --- Euler angle kinematics ---
    % Relates body angular rates [p q r] to Euler angle rates [phi_dot theta_dot psi_dot]
    euler_angles = [
        1   sin(phi)*tan(theta)   cos(phi)*tan(theta);
        0   cos(phi)             -sin(phi);
        0   sin(phi)*sec(theta)   cos(phi)*sec(theta);
        ];
    euler_rates = euler_angles * w_b;

    % --- Body angular rate dynamics (Euler's rotation equations) ---
    % wdot = J^-1 * (-omega x J*omega + m_b)
    J     = [P.Jx 0 -P.Jxz; 0 P.Jy 0; -P.Jxz 0 P.Jz];
    J_inv = inv(J);
    wdot_b = J_inv * (cross(-w_b, J*w_b) + m_b);

    % Assemble and return the 12-element state derivative vector
    out = [Pdot_ned; vgdot_b; euler_rates; wdot_b];

end
