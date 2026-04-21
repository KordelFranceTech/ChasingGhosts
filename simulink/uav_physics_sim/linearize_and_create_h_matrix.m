
P = init_quadsim_params;
P = compute_longitudinal_trim(P);
% P.delta_e0=57.296*pi/180;
P.delta_e0=1;
P.delta_a0=1;
P.delta_r0=1;
models = compute_tf_models(P);


kde=1; kpd=3; ku=4; kv=5; kw=6; kphi=7; ktheta=8; kpsi=9 ;kp=10; kq=11; kr=12; kde=1; kda=2; kdr=3; kdt=4;
[A, B] = linearize_quadsim(P)
s=tf('s')
H=ss(A,B, eye(12), zeros(12,4))
Hzpk=zpk(H);


% compute_tf_models(P);

%% G_dt2h
%% to plot H(kpd, kdt);
% linearized model / analytical open loop linear model
H_h = -1*Hzpk(kpd, kdt)
plot(step(tf(H_h),.2));
% simplified linear transfer function / numerically derived open loop linear models
G_dt2h_tf = tf(models.G_dt2h); 
G_dt2h_ss = ss(G_dt2h_tf);
G_dt2h_zpk = zpk(G_dt2h_ss);
G_dt2h_min=minreal(G_dt2h_zpk);
disp(G_dt2h_min);
t=0:.1:2;
step(models.G_dt2h)
G_dt2h_approx = 47.9/(s*(s-0.2484));
% H_h =
%      47.897
%   ------------
%   s (s-0.2484)
plot(step(G_dt2h_approx));
% compute_autopilot_gains(models, P);
% P.altitude_kp = 0.5;
% P.altitude_ki = 0.15;
% P.altitude_kd = 0.18; % 0.06
P.altitude_kp = 0.1;
P.altitude_ki = 0.012;
P.altitude_kd = 0.1;
G_hc2h = PI_rateFeedback_TF(models.G_dt2h, P.altitude_kp, P.altitude_ki, P.altitude_kd)
plot(step(G_hc2h))
plot(step(G_hc2h,out.time_s))
disp(G_hc2h)
stepinfo(G_hc2h,'RiseTimeLimits',[0 0.95])
t=0:.01:2;
plot(t,step(models.G_dt2h, t), t, step(G_dt2h_approx,t));
% plot(t/1000,step(H_h, t/1000), t,step(models.G_dt2h, t), t, step(G_dt2h_approx,t));

Gcl_roll_low =PI_rateFeedback_TF(models.G_dt2h, P.altitude_kp,P.altitude_ki,P.altitude_kd); 
Gcl_roll_high=PI_rateFeedback_TF( H(kpd,kdt), P.altitude_kp,P.altitude_ki,P.altitude_kd); 
step(Gcl_roll_low, Gcl_roll_high) % 2 seconds



%% G_de2theta
%% to plot H(ktheta, kde);
% linearized model / analytical open loop linear model
H_theta = Hzpk(ktheta, kde)
plot(step(tf(H_theta)));
% simplified linear transfer function / numerically derived open loop linear models
G_de2theta_tf = tf(models.G_de2theta);
G_de2theta_ss = ss(G_de2theta_tf);
G_de2theta_zpk = zpk(G_de2theta_ss);
G_de2theta_min=minreal(G_de2theta_zpk);
t=0:.1:2;
step(t, G_de2theta_min)
% disp(G_de2theta_min);
% % H_theta =
% %   810.75
% %   ------
% %    s^2
G_de2theta_approx = 810.75/(s^2);
step(t, G_de2theta_approx);
% P.pitch_kp = 0.01;
% P.pitch_ki = 0.002;
% P.pitch_kd = 0.006;
P.pitch_kp = 0.012; % 0.01
P.pitch_ki = 0.002;
P.pitch_kd = 0.01;
G_thetac2theta = PI_rateFeedback_TF(models.G_de2theta, P.pitch_kp, P.pitch_ki, P.pitch_kd)
step(G_thetac2theta)
% plot(step(G_hc2h,out.time_s))
disp(G_thetac2theta)
stepinfo(G_thetac2theta,'RiseTimeLimits',[0 0.95])
t=0:.01:2;
% plot(t/1000,step(H_theta, t/1000), t,step(models.G_de2theta, t), t, step(G_de2theta_approx,t));
plot(t,step(models.G_de2theta, t), t, step(G_de2theta_approx,t));

Gcl_roll_low =PI_rateFeedback_TF(models.G_de2theta, P.pitch_kp,P.pitch_ki,P.pitch_kd); 
Gcl_roll_high=PI_rateFeedback_TF( H(ktheta,kde), P.pitch_kp,P.pitch_ki,P.pitch_kd); 
step(Gcl_roll_low, Gcl_roll_high,40) % 2 seconds


%% G_da2phi
%% to plot H(ktheta, kde);
% linearized model / analytical open loop linear model
H_phi = Hzpk(kphi, kda)
plot(step(tf(H_phi)));
% simplified linear transfer function / numerically derived open loop linear models
G_da2phi_tf = tf(models.G_da2phi);
G_da2phi_ss = ss(G_da2phi_tf);
G_da2phi_zpk = zpk(G_da2phi_ss);
G_da2phi_min=minreal(G_da2phi_zpk);
disp(G_da2phi_min);
plot(step(models.G_da2phi))
% H_phi =
%   810.75
%   ------
%    s^2
G_da2phi_approx = 810.75/(s^2);
P.roll_kp = 0.012;
P.roll_ki = 0.002;
P.roll_kd = 0.01;
G_phic2phi = PI_rateFeedback_TF(models.G_da2phi, P.roll_kp, P.roll_ki, P.roll_kd)
plot(step(G_phic2phi))
% plot(step(G_hc2h,out.time_s))
disp(G_phic2phi)
stepinfo(G_phic2phi,'RiseTimeLimits',[0 0.95])
t=0:.01:2;
% plot(t/1000,step(H_phi, t/1000), t,step(models.G_da2phi, t), t, step(G_da2phi_approx,t));
plot(t,step(models.G_da2phi, t), t, step(G_da2phi_approx,t));

Gcl_roll_low =PI_rateFeedback_TF(models.G_da2phi, P.roll_kp,P.roll_ki,P.roll_kd); 
Gcl_roll_high=PI_rateFeedback_TF( H(kphi,kda), P.roll_kp,P.roll_ki,P.roll_kd); 
step(Gcl_roll_low, Gcl_roll_high, 40) % 2 seconds


%% G_dr2psi
%% to plot H(ktheta, kde);
% linearized model / analytical open loop linear model
H_psi = Hzpk(kpsi, kdr)
plot(step(tf(H_psi)));
% simplified linear transfer function / numerically derived open loop linear models
G_dr2psi_tf = tf(models.G_dr2psi);
G_dr2psi_ss = ss(G_dr2psi_tf);
G_dr2psi_zpk = zpk(G_dr2psi_ss);
G_dr2psi_min=minreal(G_dr2psi_zpk);
disp(G_dr2psi_min);
t=0:.1:2;
step(t,models.G_dr2psi)
% H_psi =
%   42.05
%   -----
%    s^2
G_dr2psi_approx = 42.05/(s^2);
P.yaw_kp = 0.1;
P.yaw_ki = 0.012;
P.yaw_kd = 0.1;
G_psic2psi = PI_rateFeedback_TF(models.G_dr2psi, P.yaw_kp, P.yaw_ki, P.yaw_kd)
plot(step(G_psic2psi))
disp(G_psic2psi)
stepinfo(G_psic2psi,'RiseTimeLimits',[0 0.95])
t=0:.01:2;
% plot(t/1000,step(H_psi, t/1000), t,step(models.G_dr2psi, t), t, step(G_dr2psi_approx,t));
plot(t,step(models.G_dr2psi, t), t, step(G_dr2psi_approx,t));
step(models.G_dr2psi, G_dr2psi_approx,40)
x
Gcl_roll_low =PI_rateFeedback_TF(models.G_dr2psi, P.yaw_kp,P.yaw_ki,P.yaw_kd); 
Gcl_roll_high=PI_rateFeedback_TF( H(kpsi,kdr), P.yaw_kp,P.yaw_ki,P.yaw_kd); 
step(Gcl_roll_low, Gcl_roll_high, 40) % 2 seconds



%% G_theta2theta
% linearized model / analytical open loop linear model
%% to plot H(ku, ktheta);
H_vhx = Hzpk(ku, ktheta);
plot(step(tf(H_vhx)));
% simplified linear transfer function / numerically derived open loop linear models
G_theta2vhx_tf = tf(models.G_theta2vhx);
G_theta2vhx_ss = ss(G_theta2vhx_tf);
G_theta2vhx_zpk = zpk(G_theta2vhx_ss);
G_theta2vhx_min=minreal(G_theta2vhx_zpk);
disp(G_theta2vhx_min);
% t=0:.1:2;
% step(t,models.G_theta2vhx)
% H_psi =
%   42.05
%   -----
%    s^2
P.vhx_kp = 0.5;
P.vhx_ki = 0.0;
P.vhx_kd = 0.0;
G_vhxc2theta = PI_rateFeedback_TF(models.G_theta2vhx, P.vhx_kp, P.vhx_ki, P.vhx_kd)
plot(step(G_vhxc2theta))
disp(G_vhxc2theta)
stepinfo(G_vhxc2theta,'RiseTimeLimits',[0 0.95])


% T=2;
% step(G_inner, G_outer, G_approx, T);

% load_quadsim;
% models = compute_tf_models(P);
% kde=1; kpd=3; ku=4; kv=5; kw=6; kphi=7; ktheta=8; kpsi=9 ;kp=10; kq=11; kr=12; kde=1; kda=2; kdr=3; kdt=4;
% % compute_autopilot_gains(models, P);
% 
% %% Lateral Channel
% [A, B] = linearize_quadsim(P);
% s=tf('s');
% H=ss(A,B, eye(12), zeros(12,4));
% 
% G_hc2h = PI_rateFeedback_TF(models.G_dt2h, kpd, kdt)
% plot(step(G_hc2h))

% %% Open Loop response from aileron to roll
% H(kphi,kda); % Higher fidelity than models.G_da2phi % Closed Loop response from roll command to roll
% G_phic2phi = PI_rateFeedback_TF(H(kphi,kda),P.roll_kp,P.roll_ki,P.roll_kd);
% % % Gcl0 =
% % %  
% % %           28.896 (s^2 + 7.027s + 82.43)
% % %   ---------------------------------------------
% % %   (s^2 + 10.68s + 33.27) (s^2 + 4.726s + 71.77)
% % %  
% % % Continuous-time zero/pole/gain model.
% % 
% % plot(step(G_phic2phi,out.time_s));
% % 
% %% Results from e_phi_max=45*pi/180 and zeta_roll=0.9 
% % >> stepinfo(G_phic2phi,'RiseTimeLimits',[0 0.95])
% %   struct with fields:
% % 
% %          RiseTime: 0.6427
% %     TransientTime: 0.9511
% %      SettlingTime: 0.9511
% %       SettlingMin: 0.9487
% %       SettlingMax: 1.0031
% %         Overshoot: 0.5482
% %        Undershoot: 0
% %              Peak: 1.0031
% %          PeakTime: 1.3025


% %% Closed Loop response from course command to course
% G_chic2chi = PI_rateFeedback_TF(G_phic2phi*models.G_phi2chi, P.course_kp,P.course_ki,P.course_kd);
% % % Gcl0 =
% % %  
% % %                27.379 (s+0.2924) (s^2 + 7.027s + 82.43)
% % %   ------------------------------------------------------------------
% % %   (s+4.085) (s+5.317) (s^2 + 1.289s + 0.4289) (s^2 + 4.713s + 70.85)
% % %  
% % % Continuous-time zero/pole/gain model.
% % 
% % plot(step(G_chic2chi,out.time_s));
% % 
% %% Results from W_chi=30 and zeta_course=2.05 
% % >> stepinfo(G_chic2chi,'RiseTimeLimits',[0 0.95])
% %   struct with fields:
% % 
% %          RiseTime: 2.8708
% %     TransientTime: 27.1786
% %      SettlingTime: 27.1786
% %       SettlingMin: 0.9514
% %       SettlingMax: 1.0495
% %         Overshoot: 4.8786
% %        Undershoot: 0
% %              Peak: 1.0495
% %          PeakTime: 6.9871
% 
% 
% %% Longitudinal Channel
% % Open Loop response from elevator to pitch
% H(ktheta,kde); % Higher fidelity than models.G_de2theta % Closed Loop response from pitch command to pitch
% G_thetac2theta = PI_rateFeedback_TF(H(ktheta,kde),P.pitch_kp,P.pitch_ki,P.pitch_kd);
% % 
% % plot(step(G_theta2theta,out.time_s));
% % 
% %% Results from e_theta_max=30*pi/180 and zeta_pitch=0.9 
% % >> stepinfo(G_thetac2theta,'RiseTimeLimits',[0 0.95])
% %   struct with fields:
% % 
% %          RiseTime: 0.3342
% %     TransientTime: 6.0477
% %      SettlingTime: 6.0477
% %       SettlingMin: 0.7078
% %       SettlingMax: 0.9200
% %         Overshoot: 24.1918
% %        Undershoot: 0
% %              Peak: 0.9200
% %          PeakTime: 1.2669
% 
% 
% % Closed Loop response from alt command to alt
% G_altc2alt = PI_rateFeedback_TF(G_thetac2theta*models.G_theta2h, P.altitude_kp,P.altitude_ki,P.altitude_kd)
% % plot(step(G_altc2alt,out.time_s));
% % % >> stepinfo(G_altc2alt,'RiseTimeLimits',[0 0.95])
% % % ans = 
% % % 
% % %   struct with fields:
% % % 
% % %          RiseTime: 1.4004
% % %     TransientTime: 4.2101
% % %      SettlingTime: 4.2101
% % %       SettlingMin: 0.9776
% % %       SettlingMax: 1.0362
% % %         Overshoot: 0.7693
% % %        Undershoot: 0
% % %              Peak: 1.0362
% % %          PeakTime: 1.9777
% 
% % Open Loop response from throttle to airspeed
% models.G_dt2Va % Note: Can't use H(ku,kdt) because it is dominated by phugoid
% % Closed Loop response from airspeed commmand to airspeed using throttle
% G_vac2va_throttle = PI_rateFeedback_TF(models.G_dt2Va, P.airspeed_throttle_kp,P.airspeed_throttle_ki,P.airspeed_throttle_kd)
% plot(step(G_vac2va_throttle,out.time_s));
% % % >> stepinfo(G_vac2va_throttle,'RiseTimeLimits',[0 0.95])
% % % ans = 
% % % 
% % %  struct with fields:
% % % 
% % %          RiseTime: 5.5763
% % %     TransientTime: 6.6936
% % %      SettlingTime: 6.6936
% % %       SettlingMin: 0.9508
% % %       SettlingMax: 1.0084
% % %         Overshoot: 0.8391
% % %        Undershoot: 0
% % %              Peak: 1.0084
% % %          PeakTime: 11.1958
% 
% 
% % Closed Loop response from airspeed commmand to airspeed using pitch
% G_vac2va_pitch = PI_rateFeedback_TF(G_thetac2theta*models.G_theta2Va, P.airspeed_pitch_kp,P.airspeed_pitch_ki,P.airspeed_pitch_kd);
% plot(step(G_vac2va_pitch,out.time_s));
% % % >> stepinfo(G_vac2va_throttle,'RiseTimeLimits',[0 0.95])
% % % ans = 
% % % 
% % %   struct with fields:
% % % 
% % %          RiseTime: 5.4420
% % %     TransientTime: 7.7513
% % %      SettlingTime: 7.7513
% % %       SettlingMin: 0.9501
% % %       SettlingMax: 1.0012
% % %         Overshoot: 0.1225
% % %        Undershoot: 0
% % %              Peak: 1.0012
% % %          PeakTime: 16.9424
