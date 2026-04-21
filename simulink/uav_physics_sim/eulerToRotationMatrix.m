% Create a 3x3 rotation matrix from Euler angles.
% Uses ZYX rotation order:
%   Rotation 1: psi about z
%   Rotation 2: theta about y
%   Rotation 3: phi about x
%
% R = eulerToRotationMatrix(phi,theta,psi)
%       Inputs:
%         phi:   rotation about x, radians
%         theta: rotation about y, radians
%         psi:   rotation about z, radians
%       Output:
%         R:     3x3 rotation matrix (i.e. direction cosine matrix)
%
% Example: For an aircraft using standard nomenclature 
%          (phi: roll, theta: pitch, psi: yaw), the resulting rotation
%          matrix expresses the rotation from a North-East-Down coordinate
%          frame to an aircraft body coordinate frame
%
%          R_ned2b = eulerToRotationMatrix(phi,theta,psi)
%


function R = eulerToRotationMatrix(phi,theta,psi)
    R_roll = [...
              1, 0, 0;...
        0, cos(phi), sin(phi);...
        0, -sin(phi), cos(phi)]; R_pitch = [...
        cos(theta), 0, -sin(theta);... 
        0, 1, 0;...
        sin(theta), 0, cos(theta)];
        R_yaw = [...
        cos(psi), sin(psi), 0;...
        -sin(psi), cos(psi), 0;...
        0, 0, 1];
    R = R_roll*R_pitch*R_yaw;