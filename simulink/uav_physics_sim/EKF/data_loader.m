function log_data = data_loader(file_name)

data = readtable(file_name);
%%
% log_data.time_s = data.x___Timestamp*1e-6;
log_data.time_s = data.Timestamp*1e-6;


% Sensor data
log_data.accel_x_mps2 = data.AccelerometerY*9.81;
log_data.accel_y_mps2 = data.AccelerometerX*9.81;
log_data.accel_z_mps2 = -data.AccelerometerZ*9.81;
log_data.gyro_y_dps = data.GyroscopeX;
log_data.gyro_x_dps = data.GyroscopeY;
log_data.gyro_z_dps = -data.GyroscopeZ;
log_data.mag_x = data.MagnetometerX;
log_data.mag_y = data.MagnetometerY;
log_data.mag_z = data.MagnetometerZ;
log_data.pressure_pa = data.PressurePa;
log_data.temp_c = data.TemperatureC;
log_data.temp_K = data.TemperatureC + 273.15;

% Kinematics
log_data.pitch_deg = data.KinematicsPitch*180/pi;
log_data.roll_deg = data.KinematicsRoll*180/pi;
log_data.yaw_deg = -data.KinematicsYaw*180/pi;

log_data.p_dps = data.KinematicsRollVelocity*180/pi;
log_data.q_dps = data.KinematicsPitchVelocity*180/pi;
log_data.r_dps = -data.KinematicsYawVelocity*180/pi;

% Autopilot
log_data.pitch_cmd = data.PIDMasterTx_Setpoint;
log_data.roll_cmd = data.PIDMasterTy_Setpoint;
log_data.yaw_cmd = -data.PIDMasterTz_Setpoint;

log_data.p_cmd = data.PIDSlaveTy_Setpoint;
log_data.q_cmd = data.PIDSlaveTx_Setpoint;
log_data.r_cmd = -data.PIDSlaveTz_Setpoint;




