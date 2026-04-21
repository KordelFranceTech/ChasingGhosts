function P = compute_autopilot_gains(models,P)
% Compute the autopilot gains that will be used in uavsim.
%
%   P = compute_autopilot_gains(models,P)
%
%   Inputs:
%      models:  Structure containing resulting simplified tranfer function
%               models, as well as coefficients used to create the TF
%               models.
%      P:       uavsim parameter structure
%
%   Outputs:
%      P:       uavsim parameter structure containing autopilot gains
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   

    %% select gains for roll loop

        % Roll Loop Design Parameters
        e_phi_max = 45*pi/180;          % rad, Amount of roll error which causes saturation
        zeta_roll = 0.9;                % Roll loop damping coefficient
        P.K_theta_DC = 0.6637;

%         % Use described method to develop roll gains
%         % Note: 
%         %       P.delta_a_max is the max aileron deflection.
%         %       models.a_phi1 and models.a_phi2 are the linear design
%         %       model coefficients.
%         P.roll_kp = (P.delta_a_max / e_phi_max) * sign(models.a_phi2);
%         omega_n_phi = sqrt(P.roll_kp * models.a_phi2);
%         P.roll_kd = ((2 * zeta_roll * omega_n_phi) - models.a_phi1) / models.a_phi2;
%         P.roll_ki = 0;
% 
%     %% select gains for course loop
%         W_chi = 30; %30 works well       % bandwidth separating factor
%         e_chi_max = 45*pi/180;           % rad, Amount of course error which causes saturation
%         zeta_course = 2.05;                % Course loop damping coefficient
%         omega_n_chi = (1 / W_chi) * omega_n_phi;
%         P.course_kp = 2 * zeta_course * omega_n_chi * P.Va0 / P.gravity;
%         P.course_kd = 0;
%         P.course_ki = (omega_n_chi ^ 2) * P.Va0 / P.gravity;
% 
%     %% select gains for sideslip hold
%     
%         % Simulated UAV doesn't have a rudder to control
%         
%     %% select gains for the pitch loop, including DC gain
%         e_theta_max = 30 *pi/180;
%         zeta_pitch = 0.9;
%         P.pitch_kp = (P.delta_e_max / e_theta_max) * sign(models.a_theta3);
%         omega_n_theta = sqrt(models.a_theta2 + (P.pitch_kp * models.a_theta3));
%         P.pitch_ki = 0;
%         P.pitch_kd = ((2*zeta_pitch * omega_n_theta) - models.a_theta1) / models.a_theta3;
%         P.K_theta_DC = (P.pitch_kp * models.a_theta3) / (models.a_theta2 + (P.pitch_kp * models.a_theta3));
% 
    %% select gains for altitude loop
%         W_h = 150;
%         zeta_alt = 2;
%         omega_n_h = (1 / W_h) * omega_n_theta;
%         P.altitude_kp = (2 * zeta_alt * omega_n_h) / (P.K_theta_DC * P.Va0);
%         P.altitude_ki = (omega_n_h ^ 2) / (P.K_theta_DC * P.Va0);
%         P.altitude_kd = 0;
        P.altitude_kp = .5;
        P.altitude_ki = 5;
        P.altitude_kd = .1;
% 
%     %% airspeed hold using pitch
%         W_V2 = 40;
%         zeta_V2 = 1;
%         omega_n_V2 = (1 / W_V2) * omega_n_theta;
%         P.airspeed_pitch_kp = (models.a_V1 - (2 * zeta_V2 * omega_n_V2)) / (P.K_theta_DC * P.gravity);
%         P.airspeed_pitch_ki = -(omega_n_V2^2) / (P.K_theta_DC * P.gravity);
%         P.airspeed_pitch_kd = 0;
% 
%     %% airspeed hold using throttle
%         W_V = 40; %15
%         zeta_V = 1; %1.3
%         omega_n_V = (1 / W_V) * omega_n_theta;
%         P.airspeed_throttle_kp = ((2 * zeta_V * omega_n_V) - models.a_V1) / models.a_V2;
%         P.airspeed_throttle_ki = (omega_n_V^2) / models.a_V2;
%         P.airspeed_throttle_kd = 0;

end