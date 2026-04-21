function models = compute_tf_models(P)
% Compute the simplified linear transfer function models that will be used
% to analytically develop autopilot control PID gains.
%
%   models = compute_tf_models(P)
%
%   Inputs:
%      P:       uavsim paramter structure
%
%   Outputs:
%      models:  Structure containing resulting simplified tranfer function
%               models, as well as coefficients used to create the TF
%               models.  (Having access to the coefficients will be useful
%               in developing the autopilot gains.)
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   

    % Define Laplace s
    s=tf('s');
    delta_x = P.L_to_motor*cos(45*pi/180);
    delta_y = P.L_to_motor*sin(45*pi/180);

    %
    % Throttle channel coefficients and models
    %
    k_Fp = (P.rho*P.C_prop*P.S_prop*(P.k_motor^2))/(P.k_omega^2);
    h_dot_dot = ((8/P.mass)*k_Fp*(P.k_omega^2)*P.delta_t0);
    models.G_dt2h = h_dot_dot/(s^2);
    
    %
    % Aileron channel coefficients and models
    %
    dev_trim_a_a = 2*((P.delta_t0 + P.delta_a0)^2); % deviation from level trim
    dev_trim_a_b = 2*((P.delta_t0 - P.delta_a0)^2); % deviation from level trim
    phi_dot_dot = delta_y*k_Fp*(P.k_omega^2) * (dev_trim_a_a - dev_trim_a_b) / P.Jx;
    models.G_da2phi = phi_dot_dot / (s^2);

    %
    % Elevator channel coefficients and models
    %
    dev_trim_e_a = 2*((P.delta_t0 + P.delta_e0)^2); % deviation from level trim
    dev_trim_e_b = 2*((P.delta_t0 - P.delta_e0)^2); % deviation from level trim
    theta_dot_dot = delta_x*k_Fp*(P.k_omega^2) * (dev_trim_e_a - dev_trim_e_b) / P.Jy;
    models.G_de2theta = theta_dot_dot / (s^2);

    %
    % Rudder channel coefficients and models
    %
    dev_trim_r_a = 2*((P.delta_t0 + P.delta_r0)^2); % deviation from level trim
    dev_trim_r_b = 2*((P.delta_t0 - P.delta_r0)^2); % deviation from level trim
    psi_dot_dot = P.k_Tp*(P.k_omega^2)*(dev_trim_r_a - dev_trim_r_b) / P.Jz;
    models.G_dr2psi = psi_dot_dot / (s^2);

end


