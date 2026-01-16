% Debug script to verify priv_initReach_Krylov test against MATLAB
% This generates exact input/output pairs for Python tests

fprintf('=== Test 1: Basic Functionality ===\n\n');

% Setup: Create a simple linear system (matching Python test)
A = [-1, 0; 0, -2];
B = [1; 1];
C = [1, 0];  % Output matrix
sys = linearSys(A, B, [], C);

fprintf('Input:\n');
fprintf('A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('B = [%g; %g]\n', B(1), B(2));
fprintf('C = [%g, %g]\n', C(1), C(2));

% Parameters
params.tStart = 0.0;
params.tFinal = 1.0;
params.R0 = zonotope([1; 1], 0.1 * eye(2));
params.U = zonotope([0.5], 0.05 * [1]);
params.uTrans = [0.1];

fprintf('params.tStart = %g\n', params.tStart);
fprintf('params.tFinal = %g\n', params.tFinal);
fprintf('params.R0 center = [%g; %g]\n', params.R0.c(1), params.R0.c(2));
fprintf('params.U center = [%g]\n', params.U.c(1));
fprintf('params.uTrans = [%g]\n', params.uTrans(1));

% Options
options.timeStep = 0.1;
options.taylorTerms = 10;
options.krylovError = 1e-6;
options.krylovOrder = 15;
options.krylovStep = 5;

fprintf('options.timeStep = %g\n', options.timeStep);
fprintf('options.taylorTerms = %d\n', options.taylorTerms);
fprintf('options.krylovError = %g\n', options.krylovError);
fprintf('options.krylovOrder = %d\n', options.krylovOrder);
fprintf('options.krylovStep = %d\n', options.krylovStep);

% Execute - priv_initReach_Krylov is a private method
% MATLAB tests enable access by changing to the private directory
% Save current directory
currDir = pwd;
% Get CORA root directory
CORAROOT = fileparts(which('linearSys'));
CORAROOT = fileparts(CORAROOT);
CORAROOT = fileparts(CORAROOT);
% Change to private directory to enable access to private functions
privateDir = fullfile(CORAROOT, 'contDynamics', '@linearSys', 'private');
cd(privateDir);
% Now we can call the private function
[sys_out, params_out, options_out] = priv_initReach_Krylov(sys, params, options);
% Return to original directory
cd(currDir);

fprintf('\n=== Output Results ===\n');

% 1. System should have krylov field
% Check if krylov field exists (it's a dynamic property)
krylov_exists = isprop(sys_out, 'krylov') || isfield(sys_out, 'krylov');
fprintf('hasattr krylov: %d\n', krylov_exists);

if krylov_exists
    try
        krylov_val = sys_out.krylov;
        fprintf('krylov is struct: %d\n', isstruct(krylov_val));
        if isstruct(krylov_val)
            fprintf('krylov fields: %s\n', strjoin(fieldnames(krylov_val), ', '));
            
            % Check for required keys
            if isfield(krylov_val, 'Rhom_tp_prev')
                fprintf('Rhom_tp_prev exists: 1\n');
                if isa(krylov_val.Rhom_tp_prev, 'zonotope')
                    fprintf('Rhom_tp_prev center = [%.15g]\n', krylov_val.Rhom_tp_prev.c(1));
                end
            end
            if isfield(krylov_val, 'Rpar_proj') || isfield(krylov_val, 'Rpar_proj_0')
                fprintf('Rpar_proj or Rpar_proj_0 exists: 1\n');
            end
            
            % State subspaces
            if isfield(krylov_val, 'state')
                fprintf('state exists: 1\n');
                state_fields = fieldnames(krylov_val.state);
                fprintf('state fields: %s\n', strjoin(state_fields, ', '));
                if isfield(krylov_val.state, 'c_sys_proj') && ~isempty(krylov_val.state.c_sys_proj)
                    fprintf('c_sys_proj exists and is not empty: 1\n');
                end
            end
            
            % Input subspaces
            if isfield(krylov_val, 'input')
                fprintf('input exists: 1\n');
                input_fields = fieldnames(krylov_val.input);
                fprintf('input fields: %s\n', strjoin(input_fields, ', '));
            end
        end
    catch ME
        fprintf('Error accessing krylov: %s\n', ME.message);
    end
end

% 2. Options should be updated
fprintf('\noptions_out.tFinal = %g\n', options_out.tFinal);
fprintf('params_out.tFinal = %g\n', params_out.tFinal);

% Save to file
fid = fopen('initReach_Krylov_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB priv_initReach_Krylov Test Output\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'B = [%g; %g]\n', B(1), B(2));
fprintf(fid, 'C = [%g, %g]\n', C(1), C(2));
fprintf(fid, 'params.tStart = %g\n', params.tStart);
fprintf(fid, 'params.tFinal = %g\n', params.tFinal);
fprintf(fid, 'options.timeStep = %g\n', options.timeStep);
fprintf(fid, 'options.taylorTerms = %d\n', options.taylorTerms);
fprintf(fid, '\nOutput:\n');
fprintf(fid, 'hasattr krylov: %d\n', isfield(sys_out, 'krylov'));
if isfield(sys_out, 'krylov')
    fprintf(fid, 'krylov fields: %s\n', strjoin(fieldnames(sys_out.krylov), ', '));
    if isfield(sys_out.krylov, 'state')
        fprintf(fid, 'state fields: %s\n', strjoin(fieldnames(sys_out.krylov.state), ', '));
    end
    if isfield(sys_out.krylov, 'input')
        fprintf(fid, 'input fields: %s\n', strjoin(fieldnames(sys_out.krylov.input), ', '));
    end
end
fprintf(fid, 'options_out.tFinal = %g\n', options_out.tFinal);
fclose(fid);

fprintf('\nResults saved to initReach_Krylov_matlab_output.txt\n');
