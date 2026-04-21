function P = compute_longitudinal_trim(P)
% COMPUTE_LONGITUDINAL_TRIM  Find the hover trim condition numerically.
%
% Uses fminsearch to find the angle-of-attack (alpha), elevator deflection
% (delta_e), and throttle (delta_t) that produce zero net force along body
% x and z and zero pitching moment — i.e. straight-and-level hovering flight.
%
% Results are written back into P so subsequent simulation blocks start
% from a valid trim state:
%   P.theta0, P.u0, P.w0        - trim pitch angle and velocities
%   P.delta_e0, P.delta_t0, ... - trim control deflections
%
%   P = compute_longitudinal_trim(P)
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012

    % Starting guess for the optimizer
    alphaStar   = 0;  % Angle of attack, rad
    delta_eStar = 0;  % Elevator deflection
    delta_tStar = 0;  % Throttle (0-1)
    initial_guess = [alphaStar, delta_eStar, delta_tStar];

    % Search for the trim condition that minimizes the cost function.
    % fminsearch requires a single-input function, so P is captured via
    % an anonymous function closure: @(trim_test) cost_function(trim_test, P)
    trim_condition = fminsearch( ...
        @(trim_test) cost_function(trim_test, P), ...
        initial_guess, ...
        optimset('TolFun', 1e-24));

    % Extract trim values from optimizer output
    alphaStar   = trim_condition(1);
    delta_eStar = trim_condition(2);
    delta_tStar = trim_condition(3);

    % Re-evaluate cost function to recover the full trim state vector
    [Jcost, x, deltas] = cost_function(trim_condition, P);

    % Write trim state into P
    P.pn0    = x(1);   P.pe0    = x(2);   P.pd0    = x(3);
    P.u0     = x(4);   P.v0     = x(5);   P.w0     = x(6);
    P.phi0   = x(7);   P.theta0 = x(8);   P.psi0   = x(9);
    P.p0     = x(10);  P.q0     = x(11);  P.r0     = x(12);
    P.alpha0 = alphaStar;

    % Write trim control deflections into P
    P.delta_e0 = deltas(1);
    P.delta_a0 = deltas(2);
    P.delta_r0 = deltas(3);
    P.delta_t0 = deltas(4);

    % Report trim result
    if Jcost < 1e-6
        fprintf('******************************************************************\n');
        fprintf('  Trim condition found, Jcost = %e\n', Jcost);
        fprintf('  alpha=%.4f deg,  de=%.4f deg,  dt=%.4f\n', ...
                alphaStar*180/pi, delta_eStar*180/pi, delta_tStar);
        fprintf('******************************************************************\n');
    else
        fprintf('******************************************************************\n');
        fprintf('  WARNING: Valid trim NOT found, Jcost = %e\n', Jcost);
        fprintf('  alpha=%.4f deg,  de=%.4f deg,  dt=%.4f\n', ...
                alphaStar*180/pi, delta_eStar*180/pi, delta_tStar);
        fprintf('******************************************************************\n');
        error('Trim condition not found')
    end

end


function [Jcost, x, deltas] = cost_function(trim_test, P)
% Cost function for the trim optimizer.
%
% Evaluates forces and moments at the candidate trim condition and returns
% the sum of squared residuals for forces Fx, Fz, and pitching moment My.
% A penalty is added if throttle exceeds the valid [0, 1] range.

    alpha   = trim_test(1);
    delta_e = trim_test(2);
    delta_t = trim_test(3);

    % Trim assumes zero wind and zeroed lateral channels
    wind_ned = zeros(3,1);
    deltas   = [0; 0; 0; delta_t];   % [delta_e delta_a delta_r delta_t]

    % Build trim state vector (retain initial position and yaw from P)
    x = zeros(12,1);
    x(1)  = P.pn0;   % North position, m
    x(2)  = P.pe0;   % East position, m
    x(3)  = P.pd0;   % Down position, m
    x(4)  = 0;       % u (body-x velocity), m/s
    x(5)  = 0;       % v (body-y velocity), m/s
    x(6)  = 0;       % w (body-z velocity), m/s
    x(7)  = 0;       % phi (roll), rad
    x(8)  = alpha;   % theta (pitch = alpha at hover), rad
    x(9)  = P.psi0;  % psi (yaw), rad
    x(10) = 0;       % p, rad/s
    x(11) = 0;       % q, rad/s
    x(12) = 0;       % r, rad/s

    % Evaluate forces and moments at this candidate trim condition
    uu = [wind_ned; deltas; x; 0];
    f_and_m = quadsim_forces_moments(uu, P);

    % Cost is the squared sum of residuals we want to drive to zero:
    %   Fx (body x force), Fz (body z force), My (pitching moment)
    Jcost = f_and_m(1)^2 + f_and_m(3)^2 + f_and_m(5)^2;

    % Penalize throttle values outside [0, 1]
    if delta_t < 0
        Jcost = Jcost + 100*abs(delta_t);
    elseif delta_t > 1
        Jcost = Jcost + 100*(delta_t - 1);
    end

end
