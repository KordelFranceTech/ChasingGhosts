% quadsim_wind.m
%
% Generation of total wind vector in NED coordinates
%
% NOTE: This is modified from fixed-wing UAVSIM method:
%       To be more appropriate for hovering vehicles (like quads)
%       the Dryden gusting model was modified.  It is now
%       provided in NED coordinates (rather than body coordinates).
%
% Inputs:
%   Gusting in NED
%   Time
%
% Outputs:
%   Wind vector in NED coordinates
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", Princeton Univ. Press, 2012
function out = quadsim_wind(uu, P)

    % Extract variables from input vector uu
    %   uu = [gust_ned(1:3); time(1)];
    k=0+(1:3);       gust_ned=uu(k);   % Gust vector, NED coordinates, m/s
    k=k(end)+(1);    time=uu(k);       % Simulation time, s

    % Generate steady wind vector in NED coordinates
    ws_ned = [P.wind_n; P.wind_e; P.wind_d];

    % compute wind vector in the inertial frame
    %   ws_ned & gust_ned -> wind_ned
    wind_ned = ws_ned + gust_ned;

    % Compile function output
    out = [wind_ned]; % Length 3

end
