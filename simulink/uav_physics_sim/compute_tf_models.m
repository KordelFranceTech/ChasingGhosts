function models = compute_tf_models(P)
% COMPUTE_TF_MODELS  Simplified linear transfer function models for each control channel.
%
% Computes second-order (double-integrator) transfer functions that relate
% each control channel input to its primary state output, linearized about
% the trim condition stored in P. These models are used to analytically
% tune the PIR autopilot gains.
%
%   models = compute_tf_models(P)
%
% Inputs:
%   P      - parameter struct with trim condition (P.delta_t0, P.delta_e0, etc.)
%
% Outputs:
%   models - struct containing transfer functions:
%     models.G_dt2h      throttle → altitude
%     models.G_da2phi    aileron  → roll
%     models.G_de2theta  elevator → pitch
%     models.G_dr2psi    rudder   → yaw
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012

    % Define Laplace s
    s=tf('s');
    gamma = (P.Jx*P.Jz) - (P.Jxz^2);
    d_chi = tan(P.phi0) - P.phi0;
    delta_x = P.L_to_motor*cos(45*pi/180);
    delta_y = P.L_to_motor*sin(45*pi/180);

    %
    % Throttle channel coefficients and models
    %
    k_Fp = (P.rho*P.C_prop*P.S_prop*(P.k_motor^2))/(P.k_omega^2);
%     delta_t_star = sqrt((P.mass * P.gravity) / (4*P.rho*P.C_prop*P.S_prop*(P.k_motor^2)));
    h_dot_dot = ((8/P.mass)*k_Fp*(P.k_omega^2)*P.delta_t0);
    models.G_dt2h = h_dot_dot/(s^2);
    
    %
    % Aileron channel coefficients and models
    %
    dev_trim_a_a = 2*((P.delta_t0 + P.delta_a0)^2); % deviation from level trim
    dev_trim_a_b = 2*((P.delta_t0 - P.delta_a0)^2); % deviation from level trim
    phi_dot_dot = delta_y*k_Fp*(P.k_omega^2) * (dev_trim_a_a - dev_trim_a_b) / P.Jx;
    models.G_da2phi = phi_dot_dot / (s^2);
%     a_phi_a = ((P.rho*(P.Va0^2)*P.S_wing*P.b)/2);
%     a_phi_b = ((P.Jz*P.C_ell_p) + (P.Jxz*P.C_n_p)) / gamma;
%     a_phi_c = P.b / (2*P.Va0);
%     a_phi_d = ((P.Jz*P.C_ell_delta_a) + (P.Jxz*P.C_n_delta_a)) / gamma;
%     models.a_phi1 = -a_phi_a * a_phi_b * a_phi_c;
%     models.a_phi2 = a_phi_a * a_phi_d;
% 
%     models.G_da2p = models.a_phi2/(s+models.a_phi1);
%     models.G_da2phi = models.G_da2p/s;
%     models.G_phi2chi = d_chi + (P.gravity / P.Va0) * (1/s);
        
    %
    % Elevator channel coefficients and models
    %
    dev_trim_e_a = 2*((P.delta_t0 + P.delta_e0)^2); % deviation from level trim
    dev_trim_e_b = 2*((P.delta_t0 - P.delta_e0)^2); % deviation from level trim
    theta_dot_dot = delta_x*k_Fp*(P.k_omega^2) * (dev_trim_e_a - dev_trim_e_b) / P.Jy;
    models.G_de2theta = theta_dot_dot / (s^2);
%     a_theta_a = (P.rho*(P.Va0^2)*P.S_wing*P.c)/2;
%     a_theta_b = 1 / P.Jy;
%     a_theta_c = P.c / (2 * P.Va0);
%     models.a_theta1 = -a_theta_a * a_theta_b * a_theta_c * P.C_m_q;
%     models.a_theta2 = -a_theta_a * a_theta_b * P.C_m_alpha;
%     models.a_theta3 = a_theta_a * a_theta_b * P.C_m_delta_e;
%     models.G_de2q = (models.a_theta3 * s) / (s^2 + (models.a_theta1 * s) + models.a_theta2);
%     models.G_de2theta = models.G_de2q * (1/s);
%     models.G_theta2h = (P.Va0 / s);
    
    %
    % Rudder channel coefficients and models
    %
    dev_trim_r_a = 2*((P.delta_t0 + P.delta_r0)^2); % deviation from level trim
    dev_trim_r_b = 2*((P.delta_t0 - P.delta_r0)^2); % deviation from level trim
    psi_dot_dot = P.k_Tp*(P.k_omega^2)*((dev_trim_r_a - dev_trim_r_b) / P.Jz);
    models.G_dr2psi = psi_dot_dot / (s^2);

end


