% load_quadsim_olfaction.m
%
% Startup script for quadsim_olfaction.mdl.
%
% This model extends quadsim with electrochemical olfaction sensors.
% Run this script before pressing Play in Simulink.
%
% Typical workflow:
%   1. Run this script:   >> load_quadsim_olfaction
%   2. Open Simulink:     the model opens automatically
%   3. Press Play in Simulink to run the simulation
%
% To change the flight trajectory, edit get_quadsim_trajectory_commands.m.
% To change vehicle or environment parameters, edit init_quadsim_params.m.

% Open the Simulink model
open('quadsim_olfaction')

% Initialize all simulation parameters (vehicle, environment, sensors, etc.)
P = init_quadsim_params;

% Compute the hover throttle trim value analytically (thrust == weight),
% then refine it with a numerical optimizer to find the full trim condition
% (alpha, elevator, throttle) that zeroes out net forces and pitch moment.
P.delta_t0 = sqrt((P.mass*P.gravity/(4*P.rho*P.C_prop*P.S_prop*P.k_motor*P.k_motor)));
P = compute_longitudinal_trim(P);

% Build simplified linear transfer function models for each control channel.
models = compute_tf_models(P);

% Autopilot gains are tuned manually in quadsim_control.m.
% Uncomment the line below to compute gains analytically from the TF models.
% P = compute_autopilot_gains(models,P);
