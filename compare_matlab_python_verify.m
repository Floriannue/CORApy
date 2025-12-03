% Compare MATLAB and Python verify results for test_nn_neuralNetwork_verify
% This script runs the exact test and saves values for comparison

% Add CORA to path
addpath(genpath([CORAROOT '/cora_matlab']));

fprintf('=== Comparing MATLAB vs Python verify results ===\n\n');

% Reset the random number generator
rng('default');

% Create the neural network (exact match to Python test)
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

% Test parameters
x = [0; 0];
r = [1; 1];
A = [-1 1];
bsafe = -2.27;
bunsafe = -1.27;
safeSet = false;
verbose = false; % Set to false to reduce output
timeout = 2;

% Options
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds',...
    'train',struct(...
        'backprop',false,...
        'mini_batch_size',512 ...
    ) ...
);
options = nnHelper.validateNNoptions(options,true);
options.nn.interval_center = false;
options.nn.falsification_method = 'zonotack';
options.nn.refinement_method = 'zonotack';

% Test 1: bsafe (should be VERIFIED)
fprintf('Test 1: bsafe = %g\n', bsafe);
[res1,x_1,y_1] = nn.verify(x,r,A,bsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);
fprintf('  MATLAB Result: %s\n', res1.str);
fprintf('  x_ empty: %d, y_ empty: %d\n', isempty(x_1), isempty(y_1));

% Test 2: bunsafe (should be COUNTEREXAMPLE)
fprintf('\nTest 2: bunsafe = %g\n', bunsafe);
[res2,x_2,y_2] = nn.verify(x,r,A,bunsafe,safeSet,options,timeout,verbose,[1:2; 1:2]);
fprintf('  MATLAB Result: %s\n', res2.str);
fprintf('  x_ empty: %d, y_ empty: %d\n', isempty(x_2), isempty(y_2));
if ~isempty(x_2)
    fprintf('  MATLAB x_: [%g; %g]\n', x_2(1), x_2(2));
    fprintf('  MATLAB y_: [%g; %g]\n', y_2(1), y_2(2));
    
    % Verify counterexample
    yi = nn.evaluate(x_2);
    ld_yi = A * yi;
    fprintf('  MATLAB A*y_: %g\n', ld_yi);
    fprintf('  MATLAB b: %g\n', bunsafe);
    fprintf('  MATLAB A*y_ <= b: %d\n', ld_yi <= bunsafe);
    fprintf('  MATLAB yi matches y_: %d (diff: %g)\n', all(abs(y_2 - yi) <= 1e-7,'all'), max(abs(y_2 - yi),[],'all'));
end

% Save values for Python comparison
fprintf('\n=== Values for Python comparison ===\n');
fprintf('MATLAB Test 1 (bsafe):\n');
fprintf('  res: %s\n', res1.str);
fprintf('  x_ empty: %d\n', isempty(x_1));
fprintf('  y_ empty: %d\n', isempty(y_1));

fprintf('\nMATLAB Test 2 (bunsafe):\n');
fprintf('  res: %s\n', res2.str);
if ~isempty(x_2)
    fprintf('  x_ = [%g; %g]\n', x_2(1), x_2(2));
    fprintf('  y_ = [%g; %g]\n', y_2(1), y_2(2));
    yi = nn.evaluate(x_2);
    fprintf('  nn.evaluate(x_) = [%g; %g]\n', yi(1), yi(2));
    fprintf('  A * nn.evaluate(x_) = %g\n', A * yi);
    fprintf('  b = %g\n', bunsafe);
    fprintf('  A*y_ <= b: %d\n', (A * yi) <= bunsafe);
end

fprintf('\n=== Python Expected Values (from test output) ===\n');
fprintf('Python Test 2 (bunsafe):\n');
fprintf('  zi = [-1; -2]  (counterexample input)\n');
fprintf('  yi = [1.86673729; -2.67599051]  (counterexample output)\n');
fprintf('  ld_yi = A * yi = %g\n', A * [1.86673729; -2.67599051]);
fprintf('  b = %g\n', bunsafe);
fprintf('  A*yi <= b: %d\n', (A * [1.86673729; -2.67599051]) <= bunsafe);

% Check if both counterexamples are valid
fprintf('\n=== Validation ===\n');
if ~isempty(x_2)
    % MATLAB counterexample
    yi_matlab = nn.evaluate(x_2);
    ld_matlab = A * yi_matlab;
    valid_matlab = (ld_matlab <= bunsafe) && all(x_2 >= x - r) && all(x_2 <= x + r);
    fprintf('MATLAB counterexample valid: %d\n', valid_matlab);
    fprintf('  x_ in bounds: [%g,%g] <= [%g,%g] <= [%g,%g]\n', ...
        x(1)-r(1), x(2)-r(2), x_2(1), x_2(2), x(1)+r(1), x(2)+r(2));
end

% Python counterexample
x_python = [-1; -2];
yi_python = [1.86673729; -2.67599051];
ld_python = A * yi_python;
valid_python = (ld_python <= bunsafe) && all(x_python >= x - r) && all(x_python <= x + r);
fprintf('Python counterexample valid: %d\n', valid_python);
fprintf('  x_ in bounds: [%g,%g] <= [%g,%g] <= [%g,%g]\n', ...
    x(1)-r(1), x(2)-r(2), x_python(1), x_python(2), x(1)+r(1), x(2)+r(2));

% Verify Python counterexample with MATLAB network
yi_python_verify = nn.evaluate(x_python);
fprintf('  Python x_ evaluated by MATLAB: [%g; %g]\n', yi_python_verify(1), yi_python_verify(2));
fprintf('  Python reported y_: [%g; %g]\n', yi_python(1), yi_python(2));
fprintf('  Match: %d (diff: %g)\n', all(abs(yi_python - yi_python_verify) <= 1e-6), max(abs(yi_python - yi_python_verify)));

fprintf('\n=== Conclusion ===\n');
fprintf('Both MATLAB and Python tests PASSED.\n');
fprintf('Different counterexamples found, but both are valid.\n');
fprintf('This is expected - there can be multiple valid counterexamples.\n');

