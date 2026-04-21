% load_quadsim.m
%
% Initializer for quadsim.mdl.
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012

% Bring up simulink model
open('quadsim')

% Load quadcopter UAV parameters
P = init_quadsim_params;

% Compute the trim condition and set trim parameters in P
% P.delta_t0 = 0;
P.delta_t0 = sqrt((P.mass*P.gravity/(4*P.rho*P.C_prop*P.S_prop*P.k_motor*P.k_motor)));


%% Maybe?
% Compute the trim condition and set trim parameters in P
% (Uncomment when necessary)
P = compute_longitudinal_trim(P);
%% OR
% P.delta_t0=0.0;
% P.delta_e0=0.0;
% P.delta_a0=0.0;
% P.delta_r0=0.0;

% Generate linear response models to be used in autopilot development
models = compute_tf_models(P);
% 
% % Compute autopilot gains
% P = compute_autopilot_gains(models,P);