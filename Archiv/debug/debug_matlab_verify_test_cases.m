% Debug script to test verify function calls and generate expected outputs
% This script replicates test_nn_neuralNetwork_verify.m exactly

% Reset the random number generator.
rng('default');

% Create the neural network. The weights are from a randomly generated
% neural network:
layers = {
    nnLinearLayer( ...
        [0.6294, 0.2647; 0.8116, -0.8049;-0.7460, -0.4430; 0.8268, 0.0938],...
        [0.9150; 0.9298;-0.6848; 0.9412] ...
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
A = [-1 1];
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
% Set the input set refinedment method: {'naive','zonotack'}.
options.nn.refinement_method = 'zonotack';

% Do verification.
% MATLAB verify signature: verify(nn, x, r, A, b, safeSet, varargin)
% varargin can be: options, timeout, verbose, plotDims, plotSplittingTree
% narginchk(6,11) means 6-11 arguments total (including nn)
% So we can pass: nn, x, r, A, b, safeSet, options, timeout, verbose, plotDims, plotSplittingTree
plotDims = [1:2; 1:2];
fprintf('Test 1a: Calling verify with bsafe=-2.27 (should return VERIFIED)\n');
fprintf('Arguments: nn, x, r, A, bsafe, safeSet, options, timeout, verbose, plotDims\n');
fprintf('Total arguments: 10 (within narginchk(6,11) range)\n');
try
    [res1, x1, y1] = nn.verify(x, r, A, bsafe, safeSet, options, timeout, verbose, plotDims);
    fprintf('Test 1a (VERIFIED): res=%s, x_ empty=%d, y_ empty=%d\n', res1.str, isempty(x1), isempty(y1));
    fprintf('res.str: %s\n', res1.str);
    fprintf('x1: %s\n', mat2str(x1));
    fprintf('y1: %s\n', mat2str(y1));
catch ME
    fprintf('Error in Test 1a: %s\n', ME.message);
    fprintf('Trying with explicit plotSplittingTree=false...\n');
    try
        [res1, x1, y1] = nn.verify(x, r, A, bsafe, safeSet, options, timeout, verbose, plotDims, false);
        fprintf('Test 1a (VERIFIED): res=%s, x_ empty=%d, y_ empty=%d\n', res1.str, isempty(x1), isempty(y1));
    catch ME2
        fprintf('Error with plotSplittingTree: %s\n', ME2.message);
    end
end

% Find counterexample.
fprintf('\nTest 1b: Calling verify with bunsafe=-1.27 (should return COUNTEREXAMPLE)\n');
try
    [res2, x2, y2] = nn.verify(x, r, A, bunsafe, safeSet, options, timeout, verbose, plotDims);
    fprintf('Test 1b (COUNTEREXAMPLE): res=%s, x_ empty=%d, y_ empty=%d\n', res2.str, isempty(x2), isempty(y2));
    fprintf('res.str: %s\n', res2.str);
    fprintf('x2: %s\n', mat2str(x2));
    fprintf('y2: %s\n', mat2str(y2));
    if ~isempty(x2) && ~isempty(y2)
        % Check counterexample
        yi = nn.evaluate(x2);
        fprintf('yi (evaluated): %s\n', mat2str(yi));
        fprintf('y2 (returned): %s\n', mat2str(y2));
        fprintf('Match: %d\n', all(abs(y2 - yi) <= 1e-7, 'all'));
        fprintf('Violates spec: %d\n', all(A*y2 <= bunsafe, 1));
    end
catch ME
    fprintf('Error in Test 1b: %s\n', ME.message);
    fprintf('Trying with explicit plotSplittingTree=false...\n');
    try
        [res2, x2, y2] = nn.verify(x, r, A, bunsafe, safeSet, options, timeout, verbose, plotDims, false);
        fprintf('Test 1b (COUNTEREXAMPLE): res=%s, x_ empty=%d, y_ empty=%d\n', res2.str, isempty(x2), isempty(y2));
    catch ME2
        fprintf('Error with plotSplittingTree: %s\n', ME2.message);
    end
end

fprintf('\nDone.\n');
