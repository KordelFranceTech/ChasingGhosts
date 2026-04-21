function [delta_1, delta_2, delta_3, delta_4] = mapChannelsToMotors(delta_e,delta_a,delta_r,delta_t)
% Map quadcopter channels to motors
%
% Inputs:
%    delta_e: Elevator
%    delta_a: Aileron
%    delta_r: Rudder
%    delta_t: Throttle
%
% Outputs:
%    delta_1: front right motor
%    delta_2: back left motor
%    delta_3: front left motor
%    delta_4: back right motor

    % Map channels to motors
    %     3   1    
    %       X
    %     2   4
    
    M = [
        1 -1 1 1;
        -1 1 1 1;
        1 1 -1 1;
        -1 -1 -1 1
        ];
    delta_matrix = [delta_e; delta_a; delta_r; delta_t];
    delta_p = M * delta_matrix;

    delta_1 = delta_p(1); % front right
    delta_2 = delta_p(2); % back left
    delta_3 = delta_p(3); % front left 
    delta_4 = delta_p(4); % back right
