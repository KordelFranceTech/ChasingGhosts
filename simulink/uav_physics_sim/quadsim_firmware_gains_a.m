
% compute_tf_models(P);
% 
% G_de2q_tf = tf(models.G_de2q);
% G_de2q_ss = ss(G_de2q_tf);
% G_de2q_zpk = zpk(G_de2q_ss);
% G_de2q_min=minreal(G_de2q_zpk);
% disp(G_de2q_min);

P = init_quadsim_params;
kde=1; kpd=3; ku=4; kv=5; kw=6; kphi=7; ktheta=8; kpsi=9 ;kp=10; kq=11; kr=12; kde=1; kda=2; kdr=3; kdt=4;
[A, B] = linearize_quadsim(P)
s=tf('s');
H=ss(A,B, eye(12), zeros(12,4))
Hzpk=zpk(H);

%% to plot H(Kq, kde);
% H0 = Hzpk(kq, kde)
% plot(step(tf(H0)));

%% to plot H(Kp, kda);
H1 = Hzpk(kp, kda)
plot(step(tf(H1)));