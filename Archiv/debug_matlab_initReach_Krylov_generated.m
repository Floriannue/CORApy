% Debug script to verify priv_initReach_Krylov generated test cases against MATLAB
% This generates exact input/output pairs for Python tests
%
% Test cases:
% 1. No input set test
% 2. Large system test

% Add CORA to path
addpath(genpath('cora_matlab'));

% Open output file
fid = fopen('matlab_initReach_Krylov_generated_output.txt', 'w');
fprintf(fid, 'MATLAB priv_initReach_Krylov Generated Tests Output\n');
fprintf(fid, '====================================================\n');

% Check if mp toolbox is available (required for Krylov operations)
hasMP = ~isempty(which('mp'));
fprintf(fid, 'MP toolbox available: %d\n', hasMP);

% Try to initialize mp to check if it works
mpWorks = false;
if hasMP
    try
        test_mp = mp(1.0, 34);
        mpWorks = true;
        fprintf('MP toolbox initialized successfully.\n');
    catch ME
        fprintf('Warning: MP toolbox found but initialization failed: %s\n', ME.message);
        fprintf(fid, 'MP toolbox initialization failed: %s\n', ME.message);
    end
end

fprintf(fid, 'MP toolbox works: %d\n', mpWorks);
if ~mpWorks
    fprintf(fid, 'WARNING: All Krylov methods (Jawecki, Saad, Wang) require MP toolbox.\n');
    fprintf(fid, 'Tests will fail because priv_inputSolution_Krylov is hardcoded to use Jawecki method.\n');
    fprintf(fid, 'Skipping tests that require MP toolbox.\n\n');
    fclose(fid);
    return;
end
fprintf(fid, '\n');

% Save current directory
currDir = pwd;
% Get CORA root directory
CORAROOT = fileparts(which('linearSys'));
CORAROOT = fileparts(CORAROOT);
CORAROOT = fileparts(CORAROOT);
% Change to private directory to enable access to private functions
privateDir = fullfile(CORAROOT, 'contDynamics', '@linearSys', 'private');
cd(privateDir);

%% Test 1: No input set test
fprintf(fid, 'Test 1: No input set test\n');
fprintf(fid, '-------------------------\n');
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];
sys = linearSys(A, B, [], C);

params = struct();
params.tStart = 0.0;
params.tFinal = 1.0;
params.R0 = zonotope([1; 1], 0.1 * eye(2));
params.U = zonotope(zeros(1,1), []);  % Empty input
params.uTrans = [0.0];

options = struct();
options.timeStep = 0.1;
options.taylorTerms = 4;  % Reduced from 10
options.krylovError = 1e-6;
options.krylovOrder = 3;  % Reduced to avoid mp dependency issues
options.krylovStep = 3;   % Reduced
options.krylovMethod = 'Saad';  % Use Saad method instead of Jawecki (which requires mp)

fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'B = [%g; %g]\n', B(1), B(2));
fprintf(fid, 'C = [%g, %g]\n', C(1), C(2));
fprintf(fid, 'params.R0 center = [%g; %g]\n', params.R0.c(1), params.R0.c(2));
fprintf(fid, 'params.U is empty: %d\n', isempty(params.U.G) || size(params.U.G, 2) == 0);
fprintf(fid, 'params.uTrans = [%g]\n', params.uTrans(1));

% Execute
[sys_out, params_out, options_out] = priv_initReach_Krylov(sys, params, options);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'hasattr krylov: %d\n', isfield(sys_out, 'krylov'));
if isfield(sys_out, 'krylov')
    fprintf(fid, 'krylov fields: %s\n', strjoin(fieldnames(sys_out.krylov), ', '));
    if isfield(sys_out.krylov, 'state')
        fprintf(fid, 'state exists: 1\n');
        fprintf(fid, 'state fields: %s\n', strjoin(fieldnames(sys_out.krylov.state), ', '));
    end
    if isfield(sys_out.krylov, 'input')
        fprintf(fid, 'input exists: %d\n', isfield(sys_out.krylov, 'input'));
        if isfield(sys_out.krylov, 'input')
            fprintf(fid, 'input fields: %s\n', strjoin(fieldnames(sys_out.krylov.input), ', '));
        end
    end
end
fprintf(fid, 'options_out.tFinal = %g\n', options_out.tFinal);
fprintf(fid, '\n');

%% Test 2: Large system test
fprintf(fid, 'Test 2: Large system test (10-dimensional)\n');
fprintf(fid, '--------------------------------------------\n');

% Setup: 10-dimensional system
rng(42);  % Set seed for reproducibility
A = randn(10, 10);
A = (A + A') / 2;  % Make symmetric
A = A - 2 * eye(10);  % Make stable

B = randn(10, 2);
C = randn(3, 10);  % 3 outputs
sys = linearSys(A, B, [], C);

params = struct();
params.tStart = 0.0;
params.tFinal = 0.5;
params.R0 = zonotope(ones(10, 1), 0.1 * eye(10));
params.U = zonotope(zeros(2, 1), 0.05 * eye(2));
params.uTrans = zeros(2, 1);

options = struct();
options.timeStep = 0.05;
options.taylorTerms = 4;  % Reduced from 10
options.krylovError = 1e-6;
options.krylovOrder = 3;  % Reduced to avoid mp dependency issues
options.krylovStep = 3;   % Reduced

fprintf(fid, 'Input:\n');
fprintf(fid, 'A shape: %dx%d\n', size(A, 1), size(A, 2));
fprintf(fid, 'B shape: %dx%d\n', size(B, 1), size(B, 2));
fprintf(fid, 'C shape: %dx%d\n', size(C, 1), size(C, 2));
fprintf(fid, 'params.R0 center shape: %dx%d\n', size(params.R0.c, 1), size(params.R0.c, 2));
fprintf(fid, 'params.U center shape: %dx%d\n', size(params.U.c, 1), size(params.U.c, 2));
fprintf(fid, 'params.tFinal = %g\n', params.tFinal);
fprintf(fid, 'options.timeStep = %g\n', options.timeStep);

% Execute
[sys_out, params_out, options_out] = priv_initReach_Krylov(sys, params, options);

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'hasattr krylov: %d\n', isfield(sys_out, 'krylov'));
if isfield(sys_out, 'krylov')
    fprintf(fid, 'krylov fields: %s\n', strjoin(fieldnames(sys_out.krylov), ', '));
    if isfield(sys_out.krylov, 'state')
        fprintf(fid, 'state exists: 1\n');
        state = sys_out.krylov.state;
        fprintf(fid, 'state fields: %s\n', strjoin(fieldnames(state), ', '));
        if isfield(state, 'c_sys_proj') && ~isempty(state.c_sys_proj)
            fprintf(fid, 'c_sys_proj exists and is not empty: 1\n');
            if isa(state.c_sys_proj, 'linearSys')
                fprintf(fid, 'c_sys_proj.A shape: %dx%d\n', size(state.c_sys_proj.A, 1), size(state.c_sys_proj.A, 2));
                fprintf(fid, 'c_sys_proj.B shape: %dx%d\n', size(state.c_sys_proj.B, 1), size(state.c_sys_proj.B, 2));
            end
        end
    end
    if isfield(sys_out.krylov, 'input')
        fprintf(fid, 'input exists: %d\n', isfield(sys_out.krylov, 'input'));
    end
end
fprintf(fid, 'options_out.tFinal = %g\n', options_out.tFinal);
fprintf(fid, '\n');

% Return to original directory
cd(currDir);

fclose(fid);
fprintf('MATLAB priv_initReach_Krylov generated tests completed. Results saved to matlab_initReach_Krylov_generated_output.txt\n');
