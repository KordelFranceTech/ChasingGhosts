% quadsim_truth_feedback.m
%
% Passes true (ground-truth) states through in the same format as the
% state estimate vector produced by quadsim_estimates.m.
%
% Use this block in place of quadsim_estimates when you want the autopilot
% to run on perfect state information — useful for isolating controller
% behavior from estimator noise during development and debugging.
%
% Inputs (flat uu vector):
%   uu = [x(1:12); wind_ned(1:3); time(1)]
%   x        - 12-state vector [pn,pe,pd, u,v,w, phi,theta,psi, p,q,r]
%   wind_ned - wind vector in NED frame, m/s
%   time     - simulation time, s
%
% Outputs:
%   out - 23-element estimate vector (same layout as quadsim_estimates output)
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012
function out = quadsim_truth_feedback(uu, P)

    % Unpack input vector
    %   uu = [x(1:12); wind_ned(1:3); time(1)]
    k=(1:12);        x=uu(k);         % 12-state vector
    k=k(end)+(1:3);  wind_ned=uu(k);  % wind vector, NED frame, m/s
    k=k(end)+(1);    time=uu(k);      % simulation time, s

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

    % Construct DCM from NED to body
    R_ned2b = eulerToRotationMatrix(phi,theta,psi);

    % Compute inertial speed components
    vg_ned = R_ned2b'*[u;v;w];
    
    % Rotate wind vector to body frame
    wind_b = R_ned2b*wind_ned;

    % Pass true states through as estimates
    pn_hat    = pn;
    pe_hat    = pe;
    h_hat     = -pd;
    phi_hat   = phi;
    theta_hat = theta;
    p_hat     = p;
    q_hat     = q;
    r_hat     = r;
    Vn_hat    = vg_ned(1);
    Ve_hat    = vg_ned(2);
    Vd_hat    = vg_ned(3);
    wn_hat    = wind_ned(1);
    we_hat    = wind_ned(2);
    psi_hat   = psi;
    
    % Compile output vector (same layout as quadsim_estimates)
    out = [...
            pn_hat;...
            pe_hat;...
            h_hat;...
            0;...       % Va_hat placeholder (no airspeed sensor on quadrotor)
            phi_hat;...
            theta_hat;...
            psi_hat;...
            p_hat;...
            q_hat;...
            r_hat;...
            Vn_hat;...
            Ve_hat;...
            Vd_hat;...
            wn_hat;...
            we_hat;...
            0; % future use
            0; % future use
            0; % future use
            0; % future use
            0; % future use
            0; % future use
            0; % future use
            0; % future use
        ]; % Length: 23
    
end 
