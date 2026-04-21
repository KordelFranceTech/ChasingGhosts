% makeVgGammaCourse.m: Create groundspeed, flight path angle (gamma) and 
%   course from a ground-relative vector in NED coordinates
%
%   [Vg gamma course] = makeVgGammaCourse(vg_ned)
%      Inputs: 
%        vg_ned:  3-element vector representing ground-relative velocity
%                 in NED coordinates
%      Outputs:
%        Vg:      Airspeed (scalar). (units are same as input vector)
%        gamma:   Vertical flight path angle, radians (+Up)
%        course:  Course (horiz. flight path angle), radians (+East-of-North)
%

function [Vg, gamma, course] = makeVgGammaCourse(vg_ned)
    Vg = sqrt(sum(vg_ned.^2)); 
    if Vg==0
        gamma=0;
        course=0;
    else
        gamma = -asin(vg_ned(3)/Vg);
        course = atan2(vg_ned(2), vg_ned(1));
    end
end 