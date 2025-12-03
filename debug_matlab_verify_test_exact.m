% Debug script to verify Python test_nn_neuralNetwork_verify_matlab_exact
% This script runs the exact same test as the Python test to compare results

% Add CORA to path
addpath(genpath([CORAROOT '/cora_matlab']));

fprintf('=== MATLAB Test: test_nn_neuralNetwork_verify (exact match to Python) ===\n\n');

% Reset the random number generator to match Python (MATLAB's 'default' seed)
rng('default');

% Create the neural network. The weights are from a randomly generated
% neural network (matching Python test exactly):
layers = {
    nnLinearLayer( ...
        [0.6294, 0.2647; 0.8116, -0.8049; -0.7460, -0.4430; 0.8268, 0.0938],...
        [0.9150; 0.9298; -0.6848; 0.9412] ...
    ); ...
    nnReLULayer(); ...
    nnLinearLayer( ...
        [0.9143, -0.1565, 0.3115, 0.3575; -0.0292, 0.8315, -0.9286, 0.5155;  0.6006, 0.5844, 0.6983, 0.4863; -0.7162, 0.9190, 0.8680, -0.2155], ...
        [0.3110; -0.6576; 0.4121; -0.9363] ...
    ); ...
    nnReLULayer(); ...
    nnLinearLayer( ...
        [-0.4462, -0.8057, 0.3897, 0.9004; -0.9077, 0.6469, -0.3658, -0.9311], ...
        [-0.1225; -0.2369] ...
    ); ...
};
nn = neuralNetwork(layers);

% Specify initial set.
x = [0; 0]; % center
r = [1; 1]; % radius

% Specify unsafe set specification.
A = [-1 1]; % Shape: (1, 2) in Python
bsafe = -2.27;
bunsafe = -1.27;
safeSet = false;

% Verbose verification output.
verbose = true;
% Set a timeout of 2s.
timeout = 2;

% Create evaluation options.
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds',...
    'train',struct(...
        'backprop',false,...
        'mini_batch_size',512 ...
    ) ...
);
% Set default training parameters
options = nnHelper.validateNNoptions(options,true);
options.nn.interval_center = false;

% Set the falsification method: {'fgsm','center','zonotack'}.
options.nn.falsification_method = 'zonotack';
% Set the input set refinement method: {'naive','zonotack'}.
options.nn.refinement_method = 'zonotack';

fprintf('Test Configuration:\n');
fprintf('  Network: 2 inputs, 2 outputs, 3 layers with ReLU\n');
fprintf('  x: [%g; %g]\n', x(1), x(2));
fprintf('  r: [%g; %g]\n', r(1), r(2));
fprintf('  A: [%g %g]\n', A(1), A(2));
fprintf('  bsafe: %g\n', bsafe);
fprintf('  bunsafe: %g\n', bunsafe);
fprintf('  safeSet: %d\n', safeSet);
fprintf('  falsification_method: %s\n', options.nn.falsification_method);
fprintf('  refinement_method: %s\n', options.nn.refinement_method);
fprintf('  timeout: %g\n', timeout);
fprintf('\n');

% Test 1: Do verification - should return VERIFIED
fprintf('=== Test 1: Verification with bsafe = %g ===\n', bsafe);
fprintf('Expected: VERIFIED, empty x_, empty y_\n');
[res1,x_1,y_1] = nn.verify(x,r,A,bsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);

fprintf('\nResults:\n');
fprintf('  res.str: %s\n', res1.str);
fprintf('  x_ empty: %d\n', isempty(x_1));
fprintf('  y_ empty: %d\n', isempty(y_1));
if ~isempty(x_1)
    fprintf('  x_ value: [%g; %g]\n', x_1(1), x_1(2));
end
if ~isempty(y_1)
    fprintf('  y_ value: [%g; %g]\n', y_1(1), y_1(2));
end

% Check assertion
test1_passed = strcmp(res1.str,'VERIFIED') && isempty(x_1) && isempty(y_1);
fprintf('\nTest 1 Result: %s\n', char(string(test1_passed).replace("1","PASSED").replace("0","FAILED")));
if ~test1_passed
    fprintf('  ERROR: Expected VERIFIED with empty x_ and y_\n');
end

% Test 2: Find counterexample - should return COUNTEREXAMPLE
fprintf('\n=== Test 2: Verification with bunsafe = %g ===\n', bunsafe);
fprintf('Expected: COUNTEREXAMPLE, non-empty x_, non-empty y_\n');
[res2,x_2,y_2] = nn.verify(x,r,A,bunsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);

fprintf('\nResults:\n');
fprintf('  res.str: %s\n', res2.str);
fprintf('  x_ empty: %d\n', isempty(x_2));
fprintf('  y_ empty: %d\n', isempty(y_2));
if ~isempty(x_2)
    fprintf('  x_ value: [%g; %g]\n', x_2(1), x_2(2));
end
if ~isempty(y_2)
    fprintf('  y_ value: [%g; %g]\n', y_2(1), y_2(2));
end

% Check counterexample validity
if ~isempty(x_2) && ~isempty(y_2)
    % Compute output of the neural network
    yi = nn.evaluate(x_2);
    % Check if output matches
    output_matches = all(abs(y_2 - yi) <= 1e-7,'all');
    % Check if output violates the specification
    if safeSet
        violates = any(A*yi >= bunsafe,1);
    else
        violates = all(A*yi <= bunsafe,1);
    end
    counterexample_valid = output_matches && violates;
    fprintf('  Counterexample valid: %d (output_matches: %d, violates: %d)\n', ...
        counterexample_valid, output_matches, violates);
else
    counterexample_valid = false;
end

% Check assertion
test2_passed = strcmp(res2.str,'COUNTEREXAMPLE') && ~isempty(x_2) && ~isempty(y_2) && counterexample_valid;
fprintf('\nTest 2 Result: %s\n', char(string(test2_passed).replace("1","PASSED").replace("0","FAILED")));
if ~test2_passed
    fprintf('  ERROR: Expected COUNTEREXAMPLE with valid counterexample\n');
end

% Summary
fprintf('\n=== Summary ===\n');
fprintf('Test 1 (bsafe): %s\n', char(string(test1_passed).replace("1","PASSED").replace("0","FAILED")));
fprintf('Test 2 (bunsafe): %s\n', char(string(test2_passed).replace("1","PASSED").replace("0","FAILED")));
if test1_passed && test2_passed
    fprintf('\nAll tests PASSED - MATLAB behavior matches expected results\n');
else
    fprintf('\nSome tests FAILED - check differences with Python implementation\n');
end

