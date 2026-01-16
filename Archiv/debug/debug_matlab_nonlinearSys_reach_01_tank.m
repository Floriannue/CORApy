% Debug script to verify nonlinearSys reach (tank example) against MATLAB
% Generates exact output for Python comparison

% Parameters --------------------------------------------------------------
dim_x = 6;
params.tFinal = 400; % final time
params.R0 = zonotope([[2; 4; 4; 2; 10; 4], 0.2*eye(dim_x)]);
params.U = zonotope([0, 0.005]);

% Reachability Settings ---------------------------------------------------
options.timeStep = 4; % time step size for reachable set computation
options.taylorTerms = 4; % number of taylor terms for reachable sets
options.zonotopeOrder = 50; % zonotope order
options.alg = 'lin';
options.tensorOrder = 2;

% System Dynamics ---------------------------------------------------------
tank = nonlinearSys(@tank6Eq); % initialize tank system

% Reachability Analysis ---------------------------------------------------
R = reach(tank, params, options);

% Numerical Evaluation ----------------------------------------------------
IH = interval(R.timeInterval.set{end});

% Output with high precision
fprintf('IH.inf = [%.15g; %.15g; %.15g; %.15g; %.15g; %.15g]\n', IH.inf);
fprintf('IH.sup = [%.15g; %.15g; %.15g; %.15g; %.15g; %.15g]\n', IH.sup);

% Save to file
fid = fopen('matlab_nonlinearSys_reach_01_tank_output.txt', 'w');
fprintf(fid, 'IH.inf = [%.15g; %.15g; %.15g; %.15g; %.15g; %.15g]\n', IH.inf);
fprintf(fid, 'IH.sup = [%.15g; %.15g; %.15g; %.15g; %.15g; %.15g]\n', IH.sup);
fclose(fid);
