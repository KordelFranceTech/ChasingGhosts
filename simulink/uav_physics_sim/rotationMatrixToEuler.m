% Create Euler angles from a 3x3 rotation matrix
% Uses ZYX rotation order:
%   Rotation 1: psi about z
%   Rotation 2: theta about y
%   Rotation 3: phi about x
%
% [phi_rad theta_rad psi_rad] = rotationMatrixToEuler(R)
%       Inputs:
%         R:     3x3 rotation matrix (i.e. direction cosine matrix)
%       Output:
%         phi:   rotation about x, radians
%         theta: rotation about y, radians
%         psi:   rotation about z, radians
%
% Example: For an aircraft using standard nomenclature 
%          (phi: roll, theta: pitch, psi: yaw), the resulting rotation
%          matrix expresses the rotation from a North-East-Down coordinate
%          frame to an aircraft body coordinate frame
%
%          [phi theta psi] = rotationMatrixToEuler(R_ned2b)
%

function [phi_rad theta_rad psi_rad] = rotationMatrixToEuler(R)
    if abs(R(1,3))==1
        phi_rad = 0;
        theta_rad = -asin(R(1,3)); 
        psi_rad = -atan2(R(2,1),R(2,2));
    else
        phi_rad = atan2(R(2,3),R(3,3));
        theta_rad = -asin(R(1,3));
        psi_rad = atan2(R(1,2),R(1,1));
    end



