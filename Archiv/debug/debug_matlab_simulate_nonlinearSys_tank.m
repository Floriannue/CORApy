% Debug script: simulate nonlinear tank system
% Generates MATLAB ground-truth for simulate outputs

clear; clc;

% System
tank = nonlinearSys(@tank6Eq);

% Parameters
params.tStart = 0;
params.tFinal = 1;
params.timeStep = 0.1;
params.x0 = [2; 4; 4; 2; 10; 4];
params.u = 0; % zero input

% Simulate
[t, x] = simulate(tank, params);

% Write outputs
fid = fopen('matlab_simulate_nonlinearSys_tank_output.txt', 'w');
fprintf(fid, 'MATLAB simulate nonlinear tank output\n');
fprintf(fid, 't = [');
fprintf(fid, '%.15g ', t);
fprintf(fid, ']\n');
fprintf(fid, 'x = [\n');
fprintf(fid, '%.15g ', x(:));
fprintf(fid, '\n]\n');
fclose(fid);
