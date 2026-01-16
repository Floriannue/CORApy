% Debug script to trace intermediate values in linReach computation
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

% Get factors
options.factor = [];
for i = 1:(options.taylorTerms+1)
    %compute initial state factor
    options.factor(i) = options.timeStep^(i)/factorial(i);    
end

fprintf('=== MATLAB linReach Debug ===\n');
fprintf('R0 center: [%s]\n', num2str(params.R0.c', '%.15f '));
fprintf('R0 generators shape: [%d %d]\n', size(params.R0.G));
fprintf('factors: [%s]\n', num2str(options.factor, '%.15f '));

% Call linReach
Rinit{1}.set = params.R0;
Rinit{1}.error = zeros(dim_x, 1);
[Rti, Rtp, dimForSplit, opts] = linReach(tank, Rinit{1}, params, options);

fprintf('\nRti type: %s\n', class(Rti));
if isa(Rti, 'zonotope')
    fprintf('Rti center: [%s]\n', num2str(Rti.c', '%.15f '));
    fprintf('Rti generators shape: [%d %d]\n', size(Rti.G));
end

fprintf('\nRtp type: %s\n', class(Rtp));
if isstruct(Rtp)
    fprintf('Rtp fields: %s\n', strjoin(fieldnames(Rtp), ', '));
    if isfield(Rtp, 'set')
        fprintf('Rtp.set type: %s\n', class(Rtp.set));
        if isa(Rtp.set, 'zonotope')
            fprintf('Rtp.set center: [%s]\n', num2str(Rtp.set.c', '%.15f '));
            fprintf('Rtp.set generators shape: [%d %d]\n', size(Rtp.set.G));
        end
    end
    if isfield(Rtp, 'error')
        fprintf('Rtp.error: [%s]\n', num2str(Rtp.error', '%.15f '));
    end
end

% Get interval hulls
IH_ti = interval(Rti);
IH_tp = interval(Rtp.set);

fprintf('\nIH_ti inf: [%s]\n', num2str(IH_ti.inf', '%.15f '));
fprintf('IH_ti sup: [%s]\n', num2str(IH_ti.sup', '%.15f '));
fprintf('\nIH_tp inf: [%s]\n', num2str(IH_tp.inf', '%.15f '));
fprintf('IH_tp sup: [%s]\n', num2str(IH_tp.sup', '%.15f '));
