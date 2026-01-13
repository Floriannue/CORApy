% Debug script to trace intermediate values in initReach computation
% MATLAB version

% Setup (same as test)
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2 * eye(dim_x));
params.U = zonotope(zeros(1, 1), 0.005 * eye(1));
params.tFinal = 4;
params.uTrans = zeros(1, 1);

options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% System
tank = nonlinearSys(@tank6Eq, 6, 1);

% options check (needed to set maxError)
[params,options] = validateOptions(tank,params,options,'FunctionName','reach');

% Compute derivatives
derivatives(tank, options);

% Get factors for reachability analysis
options.factor = [];
for i = 1:(options.taylorTerms+1)
    %compute initial state factor
    options.factor(i) = options.timeStep^(i)/factorial(i);    
end
fprintf('factors: [%s]\n', num2str(options.factor, '%.15f '));

% Compute initReach
fprintf('=== MATLAB Debug Output ===\n');
fprintf('R0 center: [%s]\n', num2str(params.R0.c', '%.15f '));
fprintf('R0 generators shape: [%d %d]\n', size(params.R0.G));
fprintf('U center: [%s]\n', num2str(params.U.c', '%.15f '));
fprintf('U generators shape: [%d %d]\n', size(params.U.G));
fprintf('timeStep: %.15f\n', options.timeStep);
fprintf('taylorTerms: %d\n', options.taylorTerms);

[Rfirst, ~] = initReach(tank, params.R0, params, options);

fprintf('\nRfirst fields: %s\n', strjoin(fieldnames(Rfirst), ', '));
fprintf('Rfirst.tp length: %d\n', length(Rfirst.tp));
fprintf('Rfirst.ti length: %d\n', length(Rfirst.ti));

if length(Rfirst.tp) > 0
    Rtp0 = Rfirst.tp{1};
    fprintf('\nRtp{1} fields: %s\n', strjoin(fieldnames(Rtp0), ', '));
    fprintf('Rtp{1}.set type: %s\n', class(Rtp0.set));
    if isa(Rtp0.set, 'zonotope')
        fprintf('Rtp{1}.set center: [%s]\n', num2str(Rtp0.set.c', '%.15f '));
        fprintf('Rtp{1}.set generators shape: [%d %d]\n', size(Rtp0.set.G));
    end
    
    % Get interval hull
    IH_tp = interval(Rtp0.set);
    fprintf('\nIH_tp inf: [%s]\n', num2str(IH_tp.inf', '%.15f '));
    fprintf('IH_tp sup: [%s]\n', num2str(IH_tp.sup', '%.15f '));
end

if length(Rfirst.ti) > 0
    Rti0 = Rfirst.ti{1};
    fprintf('\nRti{1} type: %s\n', class(Rti0));
    if isa(Rti0, 'zonotope')
        fprintf('Rti{1} center: [%s]\n', num2str(Rti0.c', '%.15f '));
    end
    IH_ti = interval(Rti0);
    fprintf('IH_ti inf: [%s]\n', num2str(IH_ti.inf', '%.15f '));
    fprintf('IH_ti sup: [%s]\n', num2str(IH_ti.sup', '%.15f '));
end
