% makeVaAlphaBeta.m: Create airspeed, angle-of-attack and sideslip from 
%   a wind-relative airspeed vector
%
%   [Va alpha beta] = makeVaAlphaBeta(v_rel_b)
%      Inputs: 
%        v_rel_b: 3-element vector representing wind-relative airspeed
%                 in body coordinates
%      Outputs:
%        Va:      Airspeed (scalar). (units are same as input vector)
%        alpha:   Angle-of-attack, radians
%        beta:    Sideslip, radians
%


function [Va, alpha, beta] = makeVaAlphaBeta(v_rel_b)
    u_rel = v_rel_b(1);
    v_rel = v_rel_b(2);
    w_rel = v_rel_b(3);
    Va = sqrt(u_rel^2 + v_rel^2 + w_rel^2); 
    alpha = atan2(w_rel,u_rel);
    if(Va>0)
        beta = asin(v_rel/Va);
    else
        beta = 0;
end 
