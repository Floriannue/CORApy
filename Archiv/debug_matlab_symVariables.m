% Debug script to verify symVariables test against MATLAB
% This generates exact input/output pairs for Python tests
%
% Add CORA to path
addpath(genpath('cora_matlab'));

fid = fopen('matlab_symVariables_output.txt', 'w');
fprintf(fid, 'MATLAB symVariables Test Output\n');
fprintf(fid, '================================\n\n');

%% Test 1: Basic symVariables without brackets
fprintf(fid, 'Test 1: Basic symVariables without brackets\n');
fprintf(fid, '-------------------------------------------\n');

sys = contDynamics('test', 3, 1, 2);
[vars, vars_der] = symVariables(sys, false);

fprintf(fid, 'Input:\n');
fprintf(fid, 'sys.nrOfDims = %d\n', sys.nrOfDims);
fprintf(fid, 'sys.nrOfInputs = %d\n', sys.nrOfInputs);
fprintf(fid, 'sys.nrOfOutputs = %d\n', sys.nrOfOutputs);
fprintf(fid, 'withBrackets = false\n');

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'vars.x length: %d\n', length(vars.x));
fprintf(fid, 'vars.x(1) = %s\n', char(vars.x(1)));
fprintf(fid, 'vars.x(2) = %s\n', char(vars.x(2)));
fprintf(fid, 'vars.x(3) = %s\n', char(vars.x(3)));
fprintf(fid, 'vars.u length: %d\n', length(vars.u));
fprintf(fid, 'vars.u(1) = %s\n', char(vars.u(1)));
fprintf(fid, 'vars.o length: %d\n', length(vars.o));
fprintf(fid, 'vars.o(1) = %s\n', char(vars.o(1)));
fprintf(fid, 'vars.o(2) = %s\n', char(vars.o(2)));
fprintf(fid, 'vars_der.x length: %d\n', length(vars_der.x));
if length(vars_der.x) > 0
    fprintf(fid, 'vars_der.x(1) = %s\n', char(vars_der.x(1)));
end
fprintf(fid, 'vars_der.u length: %d\n', length(vars_der.u));
if length(vars_der.u) > 0
    fprintf(fid, 'vars_der.u(1) = %s\n', char(vars_der.u(1)));
end
fprintf(fid, 'vars_der.o length: %d\n', length(vars_der.o));
if length(vars_der.o) > 0
    fprintf(fid, 'vars_der.o(1) = %s\n', char(vars_der.o(1)));
    if length(vars_der.o) > 1
        fprintf(fid, 'vars_der.o(2) = %s\n', char(vars_der.o(2)));
    end
end
fprintf(fid, '\n');

%% Test 2: symVariables with brackets
fprintf(fid, 'Test 2: symVariables with brackets\n');
fprintf(fid, '-----------------------------------\n');

sys2 = contDynamics('test', 2, 1, 1);
[vars2, vars_der2] = symVariables(sys2, true);

fprintf(fid, 'Input:\n');
fprintf(fid, 'sys2.nrOfDims = %d\n', sys2.nrOfDims);
fprintf(fid, 'withBrackets = true\n');

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'vars2.x length: %d\n', length(vars2.x));
fprintf(fid, 'vars2.x(1) = %s\n', char(vars2.x(1)));
fprintf(fid, 'vars2.x(2) = %s\n', char(vars2.x(2)));
fprintf(fid, 'vars_der2.x length: %d\n', length(vars_der2.x));
if length(vars_der2.x) > 0
    fprintf(fid, 'vars_der2.x(1) = %s\n', char(vars_der2.x(1)));
end
fprintf(fid, '\n');

%% Test 3: symVariables with constraints
fprintf(fid, 'Test 3: symVariables with constraints\n');
fprintf(fid, '-------------------------------------\n');
fprintf(fid, 'Note: contDynamics base class does not have nrOfConstraints property.\n');
fprintf(fid, 'This test would require a subclass like nonlinDASys.\n');
fprintf(fid, 'Skipping for now - constraints are handled by isprop check.\n');
fprintf(fid, '\n');

%% Test 4: symVariables without constraints
fprintf(fid, 'Test 4: symVariables without constraints\n');
fprintf(fid, '---------------------------------------\n');

sys4 = contDynamics('test', 2, 1, 1);
% sys4 doesn't have nrOfConstraints attribute
[vars4, vars_der4] = symVariables(sys4, false);

fprintf(fid, 'Input:\n');
fprintf(fid, 'sys4 has nrOfConstraints: %d\n', isfield(sys4, 'nrOfConstraints'));

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'vars4.y length: %d\n', length(vars4.y));
fprintf(fid, '\n');

%% Test 5: symVariables with parameters
fprintf(fid, 'Test 5: symVariables with parameters\n');
fprintf(fid, '------------------------------------\n');
fprintf(fid, 'Note: contDynamics base class does not have nrOfParam property.\n');
fprintf(fid, 'This test would require a subclass like nonlinParamSys.\n');
fprintf(fid, 'Skipping for now - parameters are handled by isprop check.\n');
fprintf(fid, '\n');

%% Test 6: symVariables without parameters
fprintf(fid, 'Test 6: symVariables without parameters\n');
fprintf(fid, '---------------------------------------\n');

sys6 = contDynamics('test', 2, 1, 1);
[vars6, vars_der6] = symVariables(sys6, false);

fprintf(fid, 'Input:\n');
fprintf(fid, 'sys6 has nrOfParam: %d\n', isprop(sys6, 'nrOfParam'));

fprintf(fid, '\nOutput:\n');
fprintf(fid, 'vars6.p length: %d\n', length(vars6.p));
fprintf(fid, '\n');

fclose(fid);
fprintf('MATLAB symVariables tests completed. Results saved to matlab_symVariables_output.txt\n');
