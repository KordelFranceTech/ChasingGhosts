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

    %
    % Velocity channel coefficients and models
    %
%     models.G_theta2vhx = s/models.G_de2theta;

%     %
%     % Aileron channel coefficients and models
%     %
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
%     %
%     % Elevator channel coefficients and models
%     %
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
%     %
%     % Throttle channel coefficients and models
%     %
%     a_V_a = (P.rho * P.C_prop * P.S_prop) / P.mass;
%     a_V_b = (P.k_motor - P.Va0);
%     a_V_c = (P.delta_t0 * (1 - 2 * P.delta_t0) * a_V_b) - (P.delta_t0 * P.Va0);
%     a_V_d = (P.rho * P.Va0 * P.S_wing) / P.mass;
%     a_V_e = (P.C_D_0 + (P.C_D_alpha * P.alpha0) + (P.C_D_delta_e * P.delta_e0));
%     models.a_V1 = (-a_V_a * a_V_c) + (a_V_d * a_V_e);
%     models.a_V2 = a_V_a * a_V_b * (P.Va0 + (2 * P.delta_t0 * a_V_b));
%     models.a_V3 = P.gravity * cos(P.theta0 - P.alpha0);
% 
%     models.G_dt2Va = (models.a_V2 / (s + models.a_V1));
%     models.G_theta2Va = (-models.a_V3 / (s + models.a_V1));
end


