% Debug script to verify priv_precompStatError test against MATLAB
% This generates exact input/output pairs for Python tests
%
% Based on test_nonlinearSys_initReach.m and how priv_precompStatError
% is called in linReach.m (line 81-82)

% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_priv_precompStatError_output.txt', 'w');
fprintf(fid, 'MATLAB priv_precompStatError Test Output\n');
fprintf(fid, '========================================\n\n');

% Save current directory
currDir = pwd;
% Get CORA root directory
CORAROOT = fileparts(which('nonlinearSys'));
CORAROOT = fileparts(CORAROOT);
CORAROOT = fileparts(CORAROOT);
% Change to private directory to enable access to private functions
privateDir = fullfile(CORAROOT, 'contDynamics', '@contDynamics', 'private');
cd(privateDir);

%% Test 1: Basic test (tensorOrder = 2, no third-order)
fprintf(fid, 'Test 1: Basic test (tensorOrder = 2)\n');
fprintf(fid, '------------------------------------\n');

% Setup exactly like test_nonlinearSys_initReach.m
dim_x = 6;  % Tank is 6D
params = struct();
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(0, 0.005);
params.tFinal = 4.0;

% Reachability settings (like test_nonlinearSys_initReach.m)
options = struct();
options.timeStep = 4.0;  % Match test
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';  % Use 'lin' algorithm
options.tensorOrder = 2;  % Less than 4, so no third-order
options.reductionTechnique = 'girard';
options.errorOrder = 10;
options.intermediateOrder = 10;
options.maxError = ones(dim_x, 1);

% System dynamics - use tank system like test_nonlinearSys_initReach.m
% This ensures derivatives are properly generated
sys = nonlinearSys(@tank6Eq);

% Options check (like test_nonlinearSys_initReach.m line 41)
[params, options] = validateOptions(sys, params, options, 'FunctionName', 'reach');

% Compute derivatives (like test_nonlinearSys_initReach.m line 44)
derivatives(sys, options);

% Set up for linearize - need Rinit
Rinit = params.R0;

% Linearize the system (this sets up linError.p)
[sys, linsys, linParams, linOptions] = linearize(sys, Rinit, params, options);

% Translate Rinit by linearization point (like linReach.m line 60)
Rdelta = Rinit + (-sys.linError.p.x);

fprintf(fid, 'Input:\n');
fprintf(fid, 'sys dimension: %d\n', sys.nrOfDims);
fprintf(fid, 'sys.linError.p.x shape: %dx%d\n', size(sys.linError.p.x, 1), size(sys.linError.p.x, 2));
fprintf(fid, 'sys.linError.p.x = [%.15g', sys.linError.p.x(1));
for i=2:min(6,length(sys.linError.p.x))
    fprintf(fid, '; %.15g', sys.linError.p.x(i));
end
fprintf(fid, ']\n');
fprintf(fid, 'sys.linError.p.u = [%.15g]\n', sys.linError.p.u(1));
fprintf(fid, 'Rdelta center shape: %dx%d\n', size(Rdelta.c, 1), size(Rdelta.c, 2));
fprintf(fid, 'Rdelta.c = [%.15g', Rdelta.c(1));
for i=2:min(6,length(Rdelta.c))
    fprintf(fid, '; %.15g', Rdelta.c(i));
end
fprintf(fid, ']\n');
fprintf(fid, 'Rdelta.G shape: %dx%d\n', size(Rdelta.G, 1), size(Rdelta.G, 2));
fprintf(fid, 'params.U center = [%.15g]\n', params.U.c(1));
fprintf(fid, 'options.tensorOrder = %d\n', options.tensorOrder);

% Execute priv_precompStatError
[H, Zdelta, errorStat, T, ind3, Zdelta3] = priv_precompStatError(sys, Rdelta, params, options);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'H is cell: %d\n', iscell(H));
if iscell(H)
    fprintf(fid, 'H length: %d\n', length(H));
    if length(H) > 0
        fprintf(fid, 'H{1} shape: %dx%d\n', size(H{1}, 1), size(H{1}, 2));
        H1_full = full(H{1});  % Convert sparse to full for fprintf
        fprintf(fid, 'H{1}(1,1) = %.15g\n', H1_full(1,1));
        if size(H1_full, 1) > 1 && size(H1_full, 2) > 1
            fprintf(fid, 'H{1}(1,2) = %.15g\n', H1_full(1,2));
        end
        if size(H1_full, 1) > 2 && size(H1_full, 2) > 2
            fprintf(fid, 'H{1}(2,2) = %.15g\n', H1_full(2,2));
        end
    end
end

fprintf(fid, 'Zdelta center shape: %dx%d\n', size(Zdelta.c, 1), size(Zdelta.c, 2));
fprintf(fid, 'Zdelta.c = [%.15g', Zdelta.c(1));
for i=2:min(6,length(Zdelta.c))
    fprintf(fid, '; %.15g', Zdelta.c(i));
end
fprintf(fid, ']\n');
fprintf(fid, 'Zdelta.G shape: %dx%d\n', size(Zdelta.G, 1), size(Zdelta.G, 2));

fprintf(fid, 'errorStat type: %s\n', class(errorStat));
if isa(errorStat, 'zonotope')
    fprintf(fid, 'errorStat center shape: %dx%d\n', size(errorStat.c, 1), size(errorStat.c, 2));
    fprintf(fid, 'errorStat.c = [%.15g', errorStat.c(1));
    for i=2:min(6,length(errorStat.c))
        fprintf(fid, '; %.15g', errorStat.c(i));
    end
    fprintf(fid, ']\n');
    fprintf(fid, 'errorStat.G shape: %dx%d\n', size(errorStat.G, 1), size(errorStat.G, 2));
    if size(errorStat.G, 2) > 0
        fprintf(fid, 'errorStat.G(1,1) = %.15g\n', errorStat.G(1,1));
    end
end

fprintf(fid, 'T is empty: %d\n', isempty(T));
fprintf(fid, 'ind3 is empty: %d\n', isempty(ind3));
fprintf(fid, 'Zdelta3 is empty: %d\n', isempty(Zdelta3));
fprintf(fid, '\n');

%% Test 2: With tensorOrder >= 4 (third-order error)
fprintf(fid, 'Test 2: With tensorOrder = 4 (third-order error)\n');
fprintf(fid, '-------------------------------------------------\n');

% Use same setup but with tensorOrder = 4
params2 = struct();
params2.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params2.U = zonotope(0, 0.005);
params2.tFinal = 4.0;

options2 = struct();
options2.timeStep = 4.0;
options2.taylorTerms = 4;
options2.zonotopeOrder = 50;
options2.alg = 'lin';
options2.tensorOrder = 3;  % = 3, so third-order should be computed (MATLAB allows 2 or 3)
options2.reductionTechnique = 'girard';
options2.errorOrder = 10;
options2.intermediateOrder = 10;
options2.errorOrder3 = 10;  % Required when tensorOrder >= 4
options2.maxError = ones(dim_x, 1);

sys2 = nonlinearSys(@tank6Eq);
[params2, options2] = validateOptions(sys2, params2, options2, 'FunctionName', 'reach');
derivatives(sys2, options2);

Rinit2 = params2.R0;
[sys2, ~, ~, ~] = linearize(sys2, Rinit2, params2, options2);
Rdelta2 = Rinit2 + (-sys2.linError.p.x);

fprintf(fid, 'Input:\n');
fprintf(fid, 'options.tensorOrder = %d\n', options2.tensorOrder);

% Execute
[H2, Zdelta2, errorStat2, T2, ind3_2, Zdelta3_2] = priv_precompStatError(sys2, Rdelta2, params2, options2);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'H2 is cell: %d\n', iscell(H2));
fprintf(fid, 'errorStat2 type: %s\n', class(errorStat2));
fprintf(fid, 'T2 is empty: %d\n', isempty(T2));
fprintf(fid, 'ind3_2 is empty: %d\n', isempty(ind3_2));
fprintf(fid, 'Zdelta3_2 is empty: %d\n', isempty(Zdelta3_2));
if ~isempty(T2)
    fprintf(fid, 'T2 is cell: %d\n', iscell(T2));
    if iscell(T2)
        fprintf(fid, 'T2 length: %d\n', length(T2));
        if length(T2) > 0 && iscell(T2{1})
            fprintf(fid, 'T2{1} length: %d\n', length(T2{1}));
        end
    end
end
fprintf(fid, '\n');

%% Test 3: With errorOrder3 specified
fprintf(fid, 'Test 3: With errorOrder3 specified\n');
fprintf(fid, '----------------------------------\n');

params3 = struct();
params3.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params3.U = zonotope(0, 0.005);
params3.tFinal = 4.0;

options3 = struct();
options3.timeStep = 4.0;
options3.taylorTerms = 4;
options3.zonotopeOrder = 50;
options3.alg = 'lin';
options3.tensorOrder = 3;  % MATLAB allows 2 or 3
options3.reductionTechnique = 'girard';
options3.errorOrder = 10;
options3.intermediateOrder = 10;
options3.errorOrder3 = 5;  % Specified
options3.maxError = ones(dim_x, 1);

sys3 = nonlinearSys(@tank6Eq);
[params3, options3] = validateOptions(sys3, params3, options3, 'FunctionName', 'reach');
derivatives(sys3, options3);

Rinit3 = params3.R0;
[sys3, ~, ~, ~] = linearize(sys3, Rinit3, params3, options3);
Rdelta3 = Rinit3 + (-sys3.linError.p.x);

fprintf(fid, 'Input:\n');
fprintf(fid, 'options.errorOrder3 = %d\n', options3.errorOrder3);

% Execute
[H3, Zdelta3, errorStat3, T3, ind3_3, Zdelta3_3] = priv_precompStatError(sys3, Rdelta3, params3, options3);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'Zdelta3_3 is empty: %d\n', isempty(Zdelta3_3));
if ~isempty(Zdelta3_3)
    fprintf(fid, 'Zdelta3_3 type: %s\n', class(Zdelta3_3));
    if isa(Zdelta3_3, 'zonotope')
        fprintf(fid, 'Zdelta3_3 center shape: %dx%d\n', size(Zdelta3_3.c, 1), size(Zdelta3_3.c, 2));
        fprintf(fid, 'Zdelta3_3.c = [%.15g', Zdelta3_3.c(1));
        for i=2:min(6,length(Zdelta3_3.c))
            fprintf(fid, '; %.15g', Zdelta3_3.c(i));
        end
        fprintf(fid, ']\n');
    end
end
fprintf(fid, '\n');

% Return to original directory
cd(currDir);

fclose(fid);
fprintf('MATLAB priv_precompStatError tests completed. Results saved to matlab_priv_precompStatError_output.txt\n');
