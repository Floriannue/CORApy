% Debug script to verify initReach_inputDependence against MATLAB
% This generates exact input/output pairs for Python tests
%
% Test Case 1: Simple 2D system with constant parameters

% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Test Case 1: Simple 2D system with constant parameters ===\n\n');

% Setup - match Python test exactly
% NOTE: initReach_inputDependence requires parametric A (matZonotope with generators)
% For a proper test, we need A to be a matZonotope with at least one generator
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];  % Add a generator to make it parametric
A = matZonotope(A_center, A_gen);
B = [0; 1];
c = [0; 0];
sys = linParamSys(A, B, c, 'constParam');

% Initial set - match Python test
% Python: Rinit = Zonotope(np.array([[0], [0]]), np.array([[0.1, 0], [0, 0.1]]))
Rinit = zonotope([0; 0], [0.1, 0; 0, 0.1]);

% Parameters
% NOTE: Python test has incorrect dimensions - fixing for MATLAB:
% - params.U should be in input dimension (1D for this system)
% - params.uTrans should be in input dimension (1D)
% - params.Uconst might be in state dimension (2D) for priv_dependentHomSol
Uconst = zonotope([0; 0], [0.05, 0; 0, 0.05]);  % 2D for priv_dependentHomSol
U = zonotope(0, 0.05);  % 1D for priv_inputSolution (input dimension)
uTrans = 0.1;  % 1D (input dimension), not [0.1; 0]
params.Uconst = Uconst;
params.U = U;  % priv_inputSolution expects params.U in input dimension
params.uTrans = uTrans;

% Options - match Python test
options.timeStep = 0.1;
options.taylorTerms = 4;
options.reductionTechnique = 'girard';
options.zonotopeOrder = 10;
options.compTimePoint = true;
options.intermediateTerms = 2;
options.originContained = false;  % priv_inputSolution expects this

fprintf('Input parameters:\n');
fprintf('  A (center) = [%.15g, %.15g; %.15g, %.15g]\n', A_center(1,1), A_center(1,2), A_center(2,1), A_center(2,2));
fprintf('  A (generators) = %d\n', A.numgens);
fprintf('  B = [%.15g; %.15g]\n', B(1), B(2));
fprintf('  c = [%.15g; %.15g]\n', c(1), c(2));
fprintf('  timeStep = %.15g\n', options.timeStep);
fprintf('  taylorTerms = %d\n', options.taylorTerms);
fprintf('  intermediateTerms = %d\n', options.intermediateTerms);
fprintf('  reductionTechnique = %s\n', options.reductionTechnique);
fprintf('  zonotopeOrder = %d\n', options.zonotopeOrder);
fprintf('\n');

% Execute function
[sys_out, Rfirst, options_out] = initReach_inputDependence(sys, Rinit, params, options);

% Extract results
Rfirst_tp = Rfirst.tp;
Rfirst_ti = Rfirst.ti;

% Get center and generators for time-point solution
tp_center = center(Rfirst_tp);
tp_G = generators(Rfirst_tp);

% Get center and generators for time-interval solution
ti_center = center(Rfirst_ti);
ti_G = generators(Rfirst_ti);

% Output results with high precision
fprintf('=== Results ===\n\n');
fprintf('sys_out.taylorTerms = %d\n', sys_out.taylorTerms);
fprintf('sys_out.stepSize = %.15g\n', sys_out.stepSize);
fprintf('\n');

fprintf('Rfirst.tp (time-point solution):\n');
fprintf('  Center:\n');
fprintf('    [%.15g\n', tp_center(1));
for i = 2:length(tp_center)
    fprintf('     %.15g\n', tp_center(i));
end
fprintf('    ]\n');
fprintf('  Generators (shape: %dx%d):\n', size(tp_G, 1), size(tp_G, 2));
fprintf('    [');
for i = 1:size(tp_G, 1)
    if i > 1
        fprintf('     ');
    end
    for j = 1:size(tp_G, 2)
        if j == 1
            fprintf('%.15g', tp_G(i, j));
        else
            fprintf(', %.15g', tp_G(i, j));
        end
    end
    if i < size(tp_G, 1)
        fprintf(';\n');
    else
        fprintf(']\n');
    end
end
fprintf('\n');

fprintf('Rfirst.ti (time-interval solution):\n');
fprintf('  Center:\n');
fprintf('    [%.15g\n', ti_center(1));
for i = 2:length(ti_center)
    fprintf('     %.15g\n', ti_center(i));
end
fprintf('    ]\n');
fprintf('  Generators (shape: %dx%d):\n', size(ti_G, 1), size(ti_G, 2));
fprintf('    [');
for i = 1:size(ti_G, 1)
    if i > 1
        fprintf('     ');
    end
    for j = 1:size(ti_G, 2)
        if j == 1
            fprintf('%.15g', ti_G(i, j));
        else
            fprintf(', %.15g', ti_G(i, j));
        end
    end
    if i < size(ti_G, 1)
        fprintf(';\n');
    else
        fprintf(']\n');
    end
end
fprintf('\n');

% Save to file
fid = fopen('initReach_inputDependence_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB initReach_inputDependence Test Output\n');
fprintf(fid, '============================================\n\n');
fprintf(fid, 'Test Case 1: Simple 2D system with constant parameters\n\n');
fprintf(fid, 'Input parameters:\n');
fprintf(fid, '  A = [%.15g, %.15g; %.15g, %.15g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, '  B = [%.15g; %.15g]\n', B(1), B(2));
fprintf(fid, '  c = [%.15g; %.15g]\n', c(1), c(2));
fprintf(fid, '  timeStep = %.15g\n', options.timeStep);
fprintf(fid, '  taylorTerms = %d\n', options.taylorTerms);
fprintf(fid, '  intermediateTerms = %d\n', options.intermediateTerms);
fprintf(fid, '  reductionTechnique = %s\n', options.reductionTechnique);
fprintf(fid, '  zonotopeOrder = %d\n', options.zonotopeOrder);
fprintf(fid, '\n');

fprintf(fid, 'Results:\n');
fprintf(fid, '  sys_out.taylorTerms = %d\n', sys_out.taylorTerms);
fprintf(fid, '  sys_out.stepSize = %.15g\n', sys_out.stepSize);
fprintf(fid, '\n');

fprintf(fid, 'Rfirst.tp center:\n');
for i = 1:length(tp_center)
    fprintf(fid, '  %.15g\n', tp_center(i));
end
fprintf(fid, '\n');

fprintf(fid, 'Rfirst.tp generators (%dx%d):\n', size(tp_G, 1), size(tp_G, 2));
for i = 1:size(tp_G, 1)
    for j = 1:size(tp_G, 2)
        if j == 1
            fprintf(fid, '  %.15g', tp_G(i, j));
        else
            fprintf(fid, ', %.15g', tp_G(i, j));
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

fprintf(fid, 'Rfirst.ti center:\n');
for i = 1:length(ti_center)
    fprintf(fid, '  %.15g\n', ti_center(i));
end
fprintf(fid, '\n');

fprintf(fid, 'Rfirst.ti generators (%dx%d):\n', size(ti_G, 1), size(ti_G, 2));
for i = 1:size(ti_G, 1)
    for j = 1:size(ti_G, 2)
        if j == 1
            fprintf(fid, '  %.15g', ti_G(i, j));
        else
            fprintf(fid, ', %.15g', ti_G(i, j));
        end
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('Results saved to initReach_inputDependence_matlab_output.txt\n');
