function [WP_n, WP_e, h_c, psi_c] = get_quadsim_trajectory_commands(time)
% GET_QUADSIM_TRAJECTORY_COMMANDS  Time-scheduled waypoint and heading commands.
%
% Called once per control step from quadsim_control.m. Returns the active
% waypoint and heading command for the given simulation time.
%
% Outputs:
%   WP_n    - Target North position (m, NED frame)
%   WP_e    - Target East position  (m, NED frame)
%   h_c     - Commanded altitude    (m, positive up)
%   psi_c   - Commanded yaw         (rad)
%
% HOW TO EDIT THE MISSION:
%   Add or modify rows in cmd_grid below. Each row takes effect at the
%   specified time and stays active until the next row's time is reached.
%   Column order: [time_s, WP_n_m, WP_e_m, altitude_m, yaw_deg]
%
%   The first row should always start at -inf so the UAV has a valid
%   command before the simulation clock starts.

% Mission definition: [time(s)  WP_n(m)  WP_e(m)  alt(m)  yaw(deg)]
cmd_grid = [ ...
       -inf      0      0      50     0   ; ...
          5      0      0      50    60   ; ...
         10      0      0      40    60   ; ...
         15     40      0      40    60   ; ...
         35     40     60      40    60   ; ...
         40     40     60      40   300   ; ...
         45     40     60     100   300   ; ...
         60    -40     80      40   300   ; ...
        100      0      0     100     0   ; ...
    ];

% Find the most recent row whose time <= current simulation time
k = find(time >= cmd_grid(:,1), 1, 'last');

% Return commands for this time step
WP_n  = cmd_grid(k,2);
WP_e  = cmd_grid(k,3);
h_c   = cmd_grid(k,4);
psi_c = cmd_grid(k,5) * pi/180;  % convert deg to rad
