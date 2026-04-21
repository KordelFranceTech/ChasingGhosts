
%% Prepare quadsim with desired parameters

% Make sure:
%   - Wind/gusting switch is turned ON
%   - Estimates feedback is turned ON
%   - Set simulation maximum time to 140 seconds
load_quadsim
P.wind_n = -1;            % m/s
P.wind_e =  3;            % m/s
P.prop_1_omega_bias = -5; % rad/s
P.prop_2_omega_bias =  5; % rad/s
P.prop_3_omega_bias = -2; % rad/s 
P.prop_4_omega_bias = 10; % rad/s

%% Run quadsim (Same as pressing 'Play' in Simulink)

sim('quadsim') % Rename simulink file if necessary!

%% Make plots
%  Makes TWO figures: 111 & 112

% Extract waypoints
WP_n=0*out.time_s;
WP_e=0*out.time_s;
for k=1:length(out.time_s)
    [WP_n(k), WP_e(k)]=get_quadsim_trajectory_commands(out.time_s(k));
end

% Specify x limits of plots
xlimits = [0 140];

% Specify font size, if desired
fontsize=11;

% No need to change anything further down
figure(111)
clf

nRows=4;

subplot(2,2,1);
h=plot(WP_e,WP_n,'ko--',out.east_gps_m,out.north_gps_m,'r.',out.east_est_m,out.north_est_m,'g',out.east_m,out.north_m,'b');
set(h(3),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('East, m'); ylabel('North, m'); axis equal; grid on;
legend('Waypoints','GPS','EKF','Truth','Location','SouthWest')

row=3; col=1; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.alt_cmd_m,'r--',out.time_s,out.alt_baro_m,'g',out.time_s,out.alt_m,'b');
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Alt, m'); grid on
ylim([0 120])
xlim(xlimits)

row=4; col=1; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.alt_rate_est_mps,'g',out.time_s,out.alt_rate_mps,'b');
set(h(1),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Alt Rate, m/s'); grid on
xlim(xlimits)

row=1; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.horz_speed_cmd_mps,'r--',out.time_s,out.horz_speed_est_mps,'g',out.time_s,out.horz_speed_mps,'b');
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Horz. Speed, m/s'); grid on
xlim(xlimits)
title('Note: Course est. may diverge at low horz. speed')

row=2; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.course_cmd_deg,'r.',out.time_s,out.course_est_deg,'g.',out.time_s,out.course_deg,'b.','markersize',2);
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Course, deg'); grid on
xlim(xlimits)

row=3; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.yaw_cmd_deg,'r--',out.time_s,out.yaw_est_deg,'g',out.time_s,out.yaw_deg,'b');
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Yaw, deg'); grid on
xlim(xlimits)
legend('Cmd','Est','Truth')

row=4; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.omega_1_rps,out.time_s,out.omega_2_rps,out.time_s,out.omega_3_rps,out.time_s,out.omega_4_rps);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Prop Speed, rad/s'); grid on
xlim(xlimits)

pause(1);

figure(112)
clf

nRows=3;

row=1; col=1; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.pitch_cmd_deg,'r--',out.time_s,out.pitch_est_deg,'g',out.time_s,out.pitch_deg,'b');
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Pitch, deg'); grid on
ylim(1.2*P.theta_max*180/pi*[-1 1])
xlim(xlimits)

row=2; col=1; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.roll_cmd_deg,'r--',out.time_s,out.roll_est_deg,'g',out.time_s,out.roll_deg,'b');
set(h(2),'color',[0 .5 0])
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Roll, deg'); grid on
ylim(1.2*P.phi_max*180/pi*[-1 1])
xlim(xlimits)
legend('Cmd','Est','Truth')

row=3; col=1; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.wind_north_mps,out.time_s,out.wind_east_mps,out.time_s,out.wind_down_mps);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Wind, m/s'); grid on
xlim(xlimits)
legend('N','E','D')

nRows=4;

row=1; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.delta_1,out.time_s,out.delta_2,out.time_s,out.delta_3,out.time_s,out.delta_4);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Motor Signals'); grid on
ylim([0 1])
xlim(xlimits)

row=2; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.delta_1,out.time_s,out.delta_2,out.time_s,out.delta_3,out.time_s,out.delta_4);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Motor Signals'); grid on
ylim(P.delta_t0+.05*[-1 1])
xlim(xlimits)

row=3; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.delta_e,out.time_s,out.delta_a,out.time_s,out.delta_r);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('EAR Channels'); grid on
ylim(1.5e-2*[-1 1])
xlim(xlimits)
legend('de','da','dr')

row=4; col=2; subplot(nRows,2,2*(row-1)+col);
h=plot(out.time_s,out.delta_t);
set(gca,'fontsize',fontsize,'fontweight','bold');
xlabel('Time, s'); ylabel('Throttle'); grid on
xlim(xlimits)

