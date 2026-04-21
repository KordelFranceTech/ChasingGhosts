% quadsim_display_olfaction.m
%
% Visualization of quadsim variables with olfaction data
%
% Inputs:
%   Various 
%
% Outputs:
%   Creates a visualization in Figure 461
%
% Adapted from Beard & McClain, "Small Unmanned Aircraft: Theory and
% Practice", RWBeard & TWMcClain, Princeton Univ. Press, 2012
%   
function quadsim_display_olfaction(uu,P)

    % Plotting flags
    show_text = 1;
    plot_commands = 1;
    plot_estimates= 1;
    
    % Extract variables from input vector uu
    %   uu = [x(1:12); f_and_m(1:6); wind_ned(1:3); ap_cmds(1:9); estimates(1:23); meas(1:18); time(1)];
    k=1:12;          x=uu(k);         % States
    k=k(end)+(1:6);  f_and_m=uu(k);   % Forces and Moments, body
    k=k(end)+(1:3);  wind_ned=uu(k);  % Wind vector, ned, m/s
    k=k(end)+(1:4);  deltas=uu(k);    % Control commands
    k=k(end)+(1:9);  ap_cmds=uu(k);   % Autopilot commands
    k=k(end)+(1:23); estimates=uu(k); % Autopilot state estimates
    k=k(end)+(1:18); meas=uu(k);      % Measurements
    k=k(end)+(1);    time=uu(k);      % Simulation time, s
%     k=k(end)+(1:3);  olf=uu(k);       % Olfaction estimates

    % Extract state variables from x
    pn    = x(1);   % North position, m
    pe    = x(2);   % East position, m
    pd    = x(3);   % Down position, m
    u     = x(4);   % body-x groundspeed component, m/s
    v     = x(5);   % body-y groundspeed component, m/s
    w     = x(6);   % body-z groundspeed component, m/s
    phi   = x(7);   % EulerAngle: roll, rad
    theta = x(8);   % EulerAngle: pitch, rad
    psi   = x(9);   % EulerAngle: yaw, rad
    p     = x(10);  % body rate about x, rad/s
    q     = x(11);  % body rate about y, rad/s
    r     = x(12);  % body rate about z, rad/s

    % Extract variables from estimates
    pn_hat       = estimates(1);  % inertial North position, m
    pe_hat       = estimates(2);  % inertial East position, m
    h_hat        = estimates(3);  % altitude, m
    %Va_hat       = estimates(4);  % airspeed, m/s
    phi_hat      = estimates(5);  % roll angle, rad
    theta_hat    = estimates(6);  % pitch angle, rad
    psi_hat      = estimates(7);  % yaw angle, rad
    p_hat        = estimates(8);  % body frame roll rate, rad/s
    q_hat        = estimates(9);  % body frame pitch rate, rad/s
    r_hat        = estimates(10); % body frame yaw rate, rad/s
    Vn_hat       = estimates(11); % north speed, m/s
    Ve_hat       = estimates(12); % east speed, m/s
    Vd_hat       = estimates(13); % downward speed, m/s
    wn_hat       = estimates(14); % wind North, m/s
    we_hat       = estimates(15); % wind East, m/s
    olf_hat      = estimates(16:23);
%     future_use   = estimates(20:23);

    % Extract a/p commands
    Vhorz_c   = ap_cmds(1);
    h_c       = ap_cmds(2);
    chi_c     = ap_cmds(3);
    phi_c     = ap_cmds(4);
    theta_c   = ap_cmds(5);
    psi_c     = ap_cmds(6);

    olfaction_a = olf_hat(1);
    olfaction_b = olf_hat(2);
    olfaction_c = olf_hat(3);
    olfaction_d = olf_hat(6);
    
    % Compute Rotation Matrices
    R_ned2b = eulerToRotationMatrix(phi,theta,psi);
    
    % Reconstruct euler angles from R_ned2b.
    [phi theta psi] = rotationMatrixToEuler(R_ned2b);

    % Rotate wind vector to body frame
    wind_b = R_ned2b*wind_ned;

    % compute horizontal groundspeed
    v_ned=R_ned2b'*[u;v;w];
    Vhorz = sqrt(v_ned(1)^2+v_ned(2)^2);

    % Map channels to motor settings
    [delta_1, delta_2, delta_3, delta_4] = mapChannelsToMotors(deltas(1),deltas(2),deltas(3),deltas(4));
    
    % define persistent variables (graphic handles)
    persistent aircraft_handle trajectory_handle ground_handle trace_handles text_handle1 text_handle2
    persistent V F patchcolors
    
    % first time function is called, initialize plot and persistent vars
    if time==0,

        figure(461)
        clf
        
        % Init aircraft drawing

        % Generate aircraft patch parameters
        [V, F, patchcolors] = aircraft_patch_params;
        
        subplot(1,3,2); hold on
        aircraft_handle = drawAircraftBody(V,F,patchcolors,...
                                               pn,pe,pd,phi,theta,psi,...
                                               []);
                                           
        trajectory_handle = plot3(pe,pn,-pd,'b-');
        ground_handle = plot3(pe,pn,0,'k');
        
        title('not drawn to scale')
        xlabel('East')
        ylabel('North')
        zlabel('Up')
        view(35,21)  % set the view angle for figure
        hold on
        grid on
        axis equal
        set(gca,'userdata',get(gca,'position'));
        
        % Init other plots (red: command, green: estimate, blue: truth)
        
        subplot(4,3,1);
        hold on
        trace_handles.h_alt_cmd = plot(nan,nan,'r--'); 
        trace_handles.h_alt_est = plot(nan,nan,'color',[0 .5 0]);
        trace_handles.h_alt     = plot(nan,nan,'b');
        hold off
        grid on;
        xlabel('Time, s');
        ylabel('Alt, m');
        
        subplot(4,3,4);
        hold on
        trace_handles.h_speed_cmd = plot(nan,nan,'r--'); 
        trace_handles.h_speed_est = plot(nan,nan,'color',[0 .5 0]);
        trace_handles.h_speed     = plot(nan,nan,'b');
        grid on;
        xlabel('Time, s');
        ylabel('VHorz, m/s');
        
        subplot(4,3,7);
        hold on
        trace_handles.h_pitch_cmd = plot(nan,nan,'r--'); 
        trace_handles.h_pitch_est = plot(nan,nan,'color',[0 .5 0]);
        trace_handles.h_pitch     = plot(nan,nan,'b');
        grid on;
        xlabel('Time, s');
        ylabel('Pitch, deg');
        
        subplot(4,3,10);
        hold on
        trace_handles.h_roll_cmd = plot(nan,nan,'r--'); 
        trace_handles.h_roll_est = plot(nan,nan,'color',[0 .5 0]);
        trace_handles.h_roll     = plot(nan,nan,'b');
        grid on;
        xlabel('Time, s');
        ylabel('Roll, deg');   


        subplot(4,3,3);
        hold on
        trace_handles.olfa_est = plot(nan,nan,'b'); 
        grid on;
        xlabel('Time, s');
        ylabel('Olfaction Sensor (Benzene), %');   

        subplot(4,3,6);
        hold on
        trace_handles.olfb_est = plot(nan,nan,'b'); 
        grid on;
        xlabel('Time, s');
        ylabel('Olfaction Sensor (Hexane), %');   

        subplot(4,3,9);
        hold on
        trace_handles.olfc_est = plot(nan,nan,'b'); 
        grid on;
        xlabel('Time, s');
        ylabel('Olfaction Sensor (C4), %');   

        subplot(4,3,12);
        hold on
        trace_handles.olfd_est = plot(nan,nan,'b'); 
        grid on;
        xlabel('Time, s');
        ylabel('Olfaction Sensor (Nitric Oxide), %');

%         text_handle1=uicontrol('parent',461,'style','text','string','', ...
%             'units','normalized','position',[0.6682+0.00    0.56      0.16    0.3650], ...
%             'horizontalalignment','left');
%         text_handle2=uicontrol('parent',461,'style','text','string','', ...
%             'units','normalized','position',[0.6682+0.16    0.56      0.16    0.3650], ...
%             'horizontalalignment','left');
    end
    
    % every other time step, update graphics data (not re-drawing)
        
    % Update aircraft on 3D plot
    h3d=get(aircraft_handle,'parent');
    drawAircraftBody(V,F,patchcolors,...
                       pn,pe,pd,phi,theta,psi,...
                       aircraft_handle);
    zoom_width_m=120;
    zlim = [0 -pd+zoom_width_m/2]; 
    zlim(1) = max(0,zlim(2)-3*zoom_width_m);
    axis(h3d,[pe-zoom_width_m/2,pe+zoom_width_m/2,pn-zoom_width_m/2,pn+zoom_width_m/2,zlim]);
    set(h3d,'position',get(h3d,'userdata'));

    % Update positions on 3D plot
    xdata=[get(trajectory_handle,'xdata') pe];
    ydata=[get(trajectory_handle,'ydata') pn];
    zdata=[get(trajectory_handle,'zdata') -pd];
    set(trajectory_handle,'xdata',xdata,'ydata',ydata,'zdata',zdata);        
    xdata=[get(ground_handle,'xdata') pe];
    ydata=[get(ground_handle,'ydata') pn];
    zdata=[get(ground_handle,'zdata') 0];
    set(ground_handle,'xdata',xdata,'ydata',ydata,'zdata',zdata);

    % Concatenate data to update truth traces
    concatDataToPlotHandle(trace_handles.h_alt,time,-pd);
    concatDataToPlotHandle(trace_handles.h_speed,time,Vhorz);
    concatDataToPlotHandle(trace_handles.h_pitch,time,theta*180/pi);
    concatDataToPlotHandle(trace_handles.h_roll,time,mod(phi*180/pi+180,360)-180);

    % Concatenate data to update estimate traces
    if(plot_estimates)
        concatDataToPlotHandle(trace_handles.h_alt_est,time,h_hat);
        concatDataToPlotHandle(trace_handles.h_speed_est,time,sqrt(Vn_hat^2+Ve_hat^2));
        concatDataToPlotHandle(trace_handles.h_pitch_est,time,theta_hat*180/pi);
        concatDataToPlotHandle(trace_handles.h_roll_est,time,mod(phi_hat*180/pi+180,360)-180);
        concatDataToPlotHandle(trace_handles.olfa_est,time,olfaction_a);
        concatDataToPlotHandle(trace_handles.olfb_est,time,olfaction_b);
        concatDataToPlotHandle(trace_handles.olfc_est,time,olfaction_c);
        concatDataToPlotHandle(trace_handles.olfd_est,time,olfaction_d);
    end
    
    % Concatenate data to update estimate traces
    if(plot_commands)
        concatDataToPlotHandle(trace_handles.h_alt_cmd,time,h_c);
        concatDataToPlotHandle(trace_handles.h_speed_cmd,time,Vhorz_c);
        concatDataToPlotHandle(trace_handles.h_pitch_cmd,time,theta_c*180/pi);
        concatDataToPlotHandle(trace_handles.h_roll_cmd,time,mod(phi_c*180/pi+180,360)-180);
    end
    
    % Display textual information
%     if show_text
%         set(text_handle1,'string', { ...
%             sprintf('time: %5.1f s',time), ...
%             sprintf(''), ...
%             sprintf('Alt: %5.1f m',-pd), ...
%             sprintf('AltRate: %5.1f m/s',-v_ned(3)), ...
%             sprintf('Vhorz: %5.1f m/s',Vhorz), ...
%             sprintf('Yaw: %5.1f deg',psi*180/pi), ...
%             sprintf(''), ...
%             sprintf('Wind'), ...
%             sprintf(' N: %4.1f m/s',wind_ned(1)), ...
%             sprintf(' E: %4.1f m/s',wind_ned(2)), ...
%             sprintf(' D: %4.1f m/s',wind_ned(3)), ...
%             });
%         m1=' '+zeros(1,20); m1(1:round(delta_1*20))=','; m1(5:10:20)=':'; m1(10:10:20)='|';
%         m2=' '+zeros(1,20); m2(1:round(delta_2*20))=','; m2(5:10:20)=':'; m2(10:10:20)='|';
%         m3=' '+zeros(1,20); m3(1:round(delta_3*20))=','; m3(5:10:20)=':'; m3(10:10:20)='|';
%         m4=' '+zeros(1,20); m4(1:round(delta_4*20))=','; m4(5:10:20)=':'; m4(10:10:20)='|';
%         
%         set(text_handle2,'string', { ...
%             sprintf('de: %8.4f (-1-1)',deltas(1)), ...
%             sprintf('da: %8.4f (-1-1)',deltas(2)), ...
%             sprintf('dr: %8.4f (-1-1)',deltas(3)), ...
%             sprintf('dt: %7.3f (0-1)',deltas(4)), ...
%             sprintf(''), ...
%             sprintf('d1: |%s',m1), ...
%             sprintf('d2: |%s',m2), ...
%             sprintf('d3: |%s',m3), ...
%             sprintf('d4: |%s',m4), ...
%             });
%     end

    % Add time to title
    persistent tNow0
    if time==0
        tNow0=now*24*3600;
    end
    set(461,'name',sprintf('Quadcopter Olfaction Simulation --  Sim Time: %.1f, Real Time: %.1f',time,now*24*3600-tNow0));
    
    % Force a drawnow each time
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenate (X,Y) or (X,Y,Z) data onto existing figure handle
function handle = concatDataToPlotHandle(handle,newX,newY,newZ)

    if ~exist('newZ')
        xdata=[get(handle,'xdata') newX];
        ydata=[get(handle,'ydata') newY];
        set(handle,'xdata',xdata,'ydata',ydata);
    else
        xdata=[get(handle,'xdata') newX];
        ydata=[get(handle,'ydata') newY];
        zdata=[get(handle,'zdata') newZ];
        set(handle,'xdata',xdata,'ydata',ydata,'zdata',zdata);
    end    

end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% drawAircraft
% return new handle if handle argument is empty
function handle = drawAircraftBody(V,F,patchcolors,...
                                     pn,pe,pd,phi,theta,psi,...
                                     handle)
    V = rotate(V', phi, theta, psi)';  % rotate aircraft
    V = translate(V', pn, pe, pd)';  % translate aircraft
    % transform vertices from NED to XYZ (for matlab rendering)
    R = [...
      0, 1, 0;...
      1, 0, 0;...
      0, 0, -1;...
      ];
    V = V*R;

    if isempty(handle),
        handle = patch('Vertices', V, 'Faces', F,...
                     'FaceVertexCData',patchcolors,...
                     'FaceColor','flat');
    else
        set(handle,'Vertices',V,'Faces',F);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function XYZ=rotate(XYZ,phi,theta,psi)
    % define rotation matrix
    R_roll = [...
          1, 0, 0;...
          0, cos(phi), -sin(phi);...
          0, sin(phi), cos(phi)];
    R_pitch = [...
          cos(theta), 0, sin(theta);...
          0, 1, 0;...
          -sin(theta), 0, cos(theta)];
    R_yaw = [...
          cos(psi), -sin(psi), 0;...
          sin(psi), cos(psi), 0;...
          0, 0, 1];
    R = R_yaw*R_pitch*R_roll;
    % rotate vertices
    XYZ = R*XYZ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% translate vertices by pn, pe, pd
function XYZ = translate(XYZ,pn,pe,pd)
	XYZ = XYZ + repmat([pn;pe;pd],1,size(XYZ,2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make parameters for drawing aircraft
function [V, F, colors] = aircraft_patch_params
    % V: Vertices
    % F: Faces
    % colors: face colors

    % parameters for drawing aircrafts  
    Ncircle=20;
    th=linspace(-pi,pi,Ncircle)';
    rArm = 1;
    rProp=.4;
    hProp=.1;
    thetaArm=45*pi/180;
    rodWidth=.05;

    nV=0;

    % Start with Center Box
    a=.3; b=.2; h=.07;
    v=[a b h; -a b h; -a -b h; a -b h; a b -h; -a b -h; -a -b -h; a -b -h]; 
    f=[1 2 3 4; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6; 5 6 7 8];
    c=[0 1 0; 0 0 1; 0 0 1; 1 1 0; .85 0 0; 1 0 0];
    V=v;
    F=f;
    colors=c;
    nV=nV+8;

    % Front right prop
    ang=thetaArm;
    V=[V; rArm*cos(ang)+rProp*cos(th) rArm*sin(ang)+rProp*sin(th) hProp+0*th];
    F(end+1,1:length(th))=[nV+(1:length(th))];
    nV=nV+length(th);
    colors(end+1,:)=[0 1 1];

    % Front left prop
    ang=-thetaArm;
    V=[V; rArm*cos(ang)+rProp*cos(th) rArm*sin(ang)+rProp*sin(th) hProp+0*th];
    F(end+1,1:length(th))=[nV+(1:length(th))];
    nV=nV+length(th);
    colors(end+1,:)=[0 1 1];

    % Back right prop
    ang=pi-thetaArm;
    V=[V; rArm*cos(ang)+rProp*cos(th) rArm*sin(ang)+rProp*sin(th) hProp+0*th];
    F(end+1,1:length(th))=[nV+(1:length(th))];
    nV=nV+length(th);
    colors(end+1,:)=.8*[1 1 1];

    % Back left prop
    ang=pi+thetaArm;
    V=[V; rArm*cos(ang)+rProp*cos(th) rArm*sin(ang)+rProp*sin(th) hProp+0*th];
    F(end+1,1:length(th))=[nV+(1:length(th))];
    nV=nV+length(th);
    colors(end+1,:)=.8*[1 1 1];

    % Twisted Rod 1
    ang=thetaArm;
    V=[V; [rArm;rArm;-rArm;-rArm]*cos(ang)+rodWidth*rand(4,1).*[1;-1;1;-1] [rArm;rArm;-rArm;-rArm]*sin(ang)+rodWidth*rand(4,1).*[1;-1;1;-1] rodWidth*rand(4,1).*[1;-1;1;-1]];
    F(end+1,1:4)=[nV+(1:4)];
    nV=nV+4;
    colors(end+1,:)=[0 0 1];

    % Twisted Rod 2
    ang=-thetaArm;
    V=[V; [rArm;rArm;-rArm;-rArm]*cos(ang)+rodWidth*rand(4,1).*[1;-1;1;-1] [rArm;rArm;-rArm;-rArm]*sin(ang)+rodWidth*rand(4,1).*[1;-1;1;-1] rodWidth*rand(4,1).*[1;-1;1;-1]];
    F(end+1,1:4)=[nV+(1:4)];
    nV=nV+4;
    colors(end+1,:)=[0 0 1];

    % Face polygons were different sizes, so replace resulting zeros with nans
    F(F==0)=nan;

    % Positive is down
    V(:,3)=-V(:,3);

    V = 10*V;   % rescale vertices
end
