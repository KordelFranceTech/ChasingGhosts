function [A, B] = linearize_quadsim(P)
% LINEARIZE_QUADSIM  Numerical linearization of the quadrotor equations of motion.
%
% Returns the state-space matrices A (12x12) and B (12x4) by numerically
% differentiating the nonlinear model about the trim condition stored in P.
%
%   [A, B] = linearize_quadsim(P)
%
% The trim condition (P.x0, P.delta0) should be computed before calling
% this function, typically via compute_longitudinal_trim(P).
%
% The linearization is performed by finite-difference perturbation:
%   A(:,i) = ( f(x0 + eps*ei, u0) - f(x0, u0) ) / eps
%   B(:,i) = ( f(x0, u0 + eps*ei) - f(x0, u0) ) / eps
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012

    % Trim state and control vectors
    x0 = [P.pn0; P.pe0; P.pd0; P.u0; P.v0; P.w0; ...
          P.phi0; P.theta0; P.psi0; P.p0; P.q0; P.r0];
    u0 = [P.delta_e0; P.delta_a0; P.delta_r0; P.delta_t0];

    % Evaluate state derivatives at the trim condition: xdot0 = f(x0, u0)
    xdot0 = eval_forces_moments_kin_dyn(x0, u0, P);

    % Build A matrix column by column via finite differences
    A = [];
    eps_perturb = 1e-8;
    for i = 1:length(x0)
        x_perturbed = x0;
        x_perturbed(i) = x_perturbed(i) + eps_perturb;
        A(:,i) = ( eval_forces_moments_kin_dyn(x_perturbed,u0,P) - xdot0 ) / eps_perturb;
    end

    % Build B matrix column by column via finite differences
    B = [];
    for i = 1:length(u0)
        u_perturbed = u0;
        u_perturbed(i) = u_perturbed(i) + eps_perturb;
        B(:,i) = ( eval_forces_moments_kin_dyn(x0,u_perturbed,P) - xdot0 ) / eps_perturb;
    end

    % Warn if linearizing about a non-zero wind condition
    if any([P.wind_n; P.wind_e; P.wind_d] ~= 0)
        disp('NOTE: Linearization performed about non-zero wind condition')
    end

end


function xdot = eval_forces_moments_kin_dyn(x, deltas, P)
% Evaluates the full nonlinear equations of motion at state x and control
% deflections deltas. Calls quadsim_forces_moments then quadsim_kin_dyn.

    wind_ned = [P.wind_n; P.wind_e; P.wind_d];
    time = 0;

    % Forces and moments: input = [wind_ned; deltas; x; time]
    uu = [wind_ned; deltas; x; time];
    f_and_m = quadsim_forces_moments(uu, P);

    % State derivatives: input = [x; f_and_m; time]
    uu = [x; f_and_m; time];
    xdot = quadsim_kin_dyn(uu, P);

end
