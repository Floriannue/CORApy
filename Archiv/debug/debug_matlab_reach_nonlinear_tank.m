% Debug script: reach for nonlinear tank (matches Python tests)
% Generates MATLAB ground-truth for reach outputs

clear; clc;

% Parameters
dim_x = 6;
params.tFinal = 400;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(0,0.005);

% Options
options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;
options.compOutputSet = false;

% System
tank = nonlinearSys(@tank6Eq);
derivatives(tank, options);
addpath([CORAROOT filesep 'models' filesep 'auxiliary' filesep tank.name]);

% Reachability
R = reach(tank, params, options);

% Extract final interval hull
IH = interval(R.timeInterval.set{end});

% Write outputs
outPath = fullfile(tempdir, 'matlab_reach_nonlinear_tank_output.txt');
fid = fopen(outPath, 'w');
if fid == -1
    error('Failed to open output file: %s', outPath);
end
fprintf(fid, 'MATLAB reach nonlinear tank output\n');
fprintf(fid, 'IH_inf = [');
fprintf(fid, '%.15g ', IH.inf);
fprintf(fid, ']\n');
fprintf(fid, 'IH_sup = [');
fprintf(fid, '%.15g ', IH.sup);
fprintf(fid, ']\n');
fclose(fid);
disp(['MATLAB reach output path: ' outPath]);
