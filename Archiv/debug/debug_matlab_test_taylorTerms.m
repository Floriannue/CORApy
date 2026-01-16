% Test with different taylorTerms values
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Testing with different taylorTerms ===\n\n');

% Create a simple matZonotope
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];
A = matZonotope(A_center, A_gen);
B = [0; 1];
c = [0; 0];
sys = linParamSys(A, B, c, 'constParam');

% Initial set
Rinit = zonotope([0; 0], [0.1, 0; 0, 0.1]);

% Parameters
Uconst = zonotope([0; 0], [0.05, 0; 0, 0.05]);
U = zonotope([0; 0], [0.05, 0; 0, 0.05]);
uTrans = [0.1; 0];
params.Uconst = Uconst;
params.U = U;
params.uTrans = uTrans;

% Options - test with taylorTerms=8 (like RLC test)
options.timeStep = 0.1;
options.taylorTerms = 8;  % Changed from 4 to 8
options.reductionTechnique = 'girard';
options.zonotopeOrder = 10;
options.compTimePoint = true;
options.intermediateTerms = 2;
options.originContained = false;

fprintf('Test with taylorTerms=8:\n');
fprintf('  intermediateTerms=%d, taylorTerms=%d\n', options.intermediateTerms, options.taylorTerms);
fprintf('  Loop range: %d:%d\n', options.intermediateTerms+1, options.taylorTerms);
fprintf('\n');

try
    [sys_out, Rfirst, options_out] = initReach_inputDependence(sys, Rinit, params, options);
    fprintf('  SUCCESS with taylorTerms=8!\n');
    fprintf('  Rfirst.tp center: [%.15g; %.15g]\n', center(Rfirst.tp));
    fprintf('  Rfirst.ti center: [%.15g; %.15g]\n', center(Rfirst.ti));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end

fprintf('\n');

% Now test with taylorTerms=4 (original test)
options.taylorTerms = 4;
fprintf('Test with taylorTerms=4:\n');
fprintf('  intermediateTerms=%d, taylorTerms=%d\n', options.intermediateTerms, options.taylorTerms);
fprintf('  Loop range: %d:%d\n', options.intermediateTerms+1, options.taylorTerms);
fprintf('\n');

try
    [sys_out2, Rfirst2, options_out2] = initReach_inputDependence(sys, Rinit, params, options);
    fprintf('  SUCCESS with taylorTerms=4!\n');
    fprintf('  Rfirst.tp center: [%.15g; %.15g]\n', center(Rfirst2.tp));
    fprintf('  Rfirst.ti center: [%.15g; %.15g]\n', center(Rfirst2.ti));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end
