close all
load_data =1;
if load_data
    log_file = 'EKF/data_ahrs.csv';
    flight_data = data_loader(log_file);
end
%%
P.tau_gyro = 0.01; % 0.05
P.tau_mag = 0.2;
P.tau_accel = 0.01;
P.sigma_noise_gyro = .015*sqrt(80)*pi/180; % rad/s
P.sigma_noise_accel = 1/1e4*P.gravity; % m/s^2

%%
tHistory = []; % Time vector
xhatHistory = []; % Estimated measurement vectors (n columns)
pHistory = []; % P diagonal elements (n columns)
residHistory = [];
residUncHistory = [];
xHistory = [];
rhoHistory = [];
dtHistory = [];

t0 = flight_data.time_s(2);
    
for i = 2:length(flight_data.time_s)
    time = flight_data.time_s(i) - t0;
    dt = flight_data.time_s(i)-flight_data.time_s(i-1);
    P.Ts = dt;
    
    % NO GPS ON FLYBRIX
    meas(1) = 0.0; % GPS North Measurement, m
    meas(2) = 0.0; % GPS East Measurement, m
    meas(3) = 0.0; % GPS Altitude Measurement, m
    meas(4) = 0.0; % GPS North Speed Measurement, m/s
    meas(5) = 0.0; % GPS East Speed Measurement, m/s
    meas(6) = 0.0; % GPS Downward Speed Measurement, m/s
    
    % Logged gyroscope measurements
    meas(7) = flight_data.gyro_x_dps(i)*pi/180; % Gyro Body Rate Meas. about x, rad/s
    meas(8) = flight_data.gyro_y_dps(i)*pi/180; % Gyro Body Rate Meas. about y, rad/s
    meas(9) = flight_data.gyro_z_dps(i)*pi/180; % Gyro Body Rate Meas. about z, rad/s
    
    % Logged accelerometer measurements
    meas(10) = flight_data.accel_x_mps2(i); % Accelerometer Meas along x, m/s/s
    meas(11) = flight_data.accel_y_mps2(i); % Accelerometer Meas along y, m/s/s
    meas(12) = flight_data.accel_z_mps2(i); % Accelerometer Meas along z, m/s/s
    
    % Barometric pressure sensor IGNORE FOR THESE EXPERIMENTS
    meas(13) = 0.0; % Static Pressure Meas., N/m^2
    meas(14) = 0.0; % Differential Pressure Meas., N/m^2
    
    % ASSUME 0 PSI FOR THESE EXPERIMENTS
    meas(15) = 0.0; % Yaw Meas. from Magnetometer, rad
    
    meas(16) = 0.0; % Future Use
    meas(17)  = 0.0; % Futrue Use
    meas(18) = 0.0; % Future Use
    
    meas(19) = time; % Time;
    
    % Run uavsim_estimates
    out = uavsim_estimates(meas,P);
    
    % Pull off desired output
    phi_hat = out(5);
    theta_hat = out(6);
    psi_hat = out(7);
    residual = [out(18) out(19) out(20)];
    xhat = [phi_hat theta_hat psi_hat];
    
    resid_x_unc = out(21);
    resid_y_unc = out(22);
    resid_z_unc = out(23);
    
    % Logged flybrix state for comparison
    roll = flight_data.roll_deg(i)*pi/180;
    pitch = flight_data.pitch_deg(i)*pi/180;
    yaw = flight_data.yaw_deg(i)*pi/180;
    x = [roll pitch yaw]';
    
    %====================
    % Retain history
    tHistory(end+1,:) = time';    
    xHistory(end+1,:) = x*180/pi';       
    xhatHistory(end+1,:) = xhat*180/pi';
    residHistory(end+1,:) = residual;
    residUncHistory(end+1,:) = [resid_x_unc resid_y_unc resid_z_unc];
    dtHistory(end+1) = dt;
end
%% Plotting
figure
subplot(3,1,1);hold on;grid on;
% plot(tHistory,xHistorty(:,1),'b')
plot(tHistory,xhatHistory(:,1),'b')
% legend({'Flybrix Est.','Roll-Pitch-Yaw EKF Est.'})
legend({'Roll-Pitch-Yaw EKF Est.'})
xlabel('Time (s)'); ylabel('Roll (deg)')
subplot(3,1,2);hold on;grid on;
% plot(tHistory,xHistory(:,2),'b')
plot(tHistory,xhatHistory(:,2),'b')
xlabel('Time (s)');ylabel('Pitch (deg)')
subplot(3,1,3);hold on;grid on;
% plot(tHistory,xHistory(:,3),'b')
plot(tHistory,xhatHistory(:,3),'b')
xlabel('Time (s)');ylabel('Yaw (deg)')


figure
ax(1)=subplot(3,1,1);hold on;grid on;
    plot(tHistory,residHistory(:,1),'r')
    plot(tHistory,residUncHistory(:,1).*[-3 3],'k')
    xlabel('Time (s)'); ylabel('Accel Resid. X (mps2)')
    legend({'Residual','Residual 3-\sigma'})
    ylim([-3 3]) % Set to appropraite limits for your data
ax(2)=subplot(3,1,2);hold on;grid on;
    plot(tHistory,residHistory(:,2),'r')
    plot(tHistory,residUncHistory(:,2).*[-3 3],'k')
    xlabel('Time (s)');ylabel('Accel Resid. Y (mps2)')
    ylim([-3 3]) % Set to appropraite limits for your data
ax(3)=subplot(3,1,3);hold on;grid on;
    plot(tHistory,residHistory(:,3),'r')
    plot(tHistory,residUncHistory(:,3).*[-3 3],'k')
    xlabel('Time (s)');ylabel('Accel Resid. Z (mps2)')   
    ylim([-3 3]) % Set to appropraite limits for your data
linkaxes(ax,'xy')
