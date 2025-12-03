% Debug script to trace zonotack attack computation
% This will help identify why Python finds out-of-bounds counterexamples

% Add CORA to path
addpath(genpath([CORAROOT '/cora_matlab']));

fprintf('=== Debug Zonotack Attack Computation ===\n\n');

% Reset random number generator
rng('default');

% Create the neural network (same as test)
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
bunsafe = -1.27;
safeSet = false;
verbose = true;
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

fprintf('Input set:\n');
fprintf('  x: [%g; %g]\n', x(1), x(2));
fprintf('  r: [%g; %g]\n', r(1), r(2));
fprintf('  bounds: [%g,%g] x [%g,%g]\n', x(1)-r(1), x(1)+r(1), x(2)-r(2), x(2)+r(2));
fprintf('\n');

% Call verify directly - it will handle everything internally
% We'll add debug output by modifying verify.m temporarily or using a wrapper
fprintf('Calling verify with zonotack method...\n');
fprintf('This will trace through the zonotack attack computation.\n\n');

% Actually, we can't easily access nested functions, so let's just run verify
% and see what counterexample it produces
[result, x_, y_] = nn.verify(x, r, A, bunsafe, safeSet, options, timeout, verbose);

fprintf('\n=== Results ===\n');
if isstruct(result)
    fprintf('Result: %s\n', result.str);
    if isfield(result, 'time')
        fprintf('Time: %.3f [s]\n', result.time);
    end
    if isfield(result, 'numVerified')
        fprintf('Verified patches: %d\n', result.numVerified);
    end
else
    fprintf('Result: %s\n', char(result));
end

if ~isempty(x_)
    fprintf('Counterexample x_: [%g; %g]\n', x_(1), x_(2));
    fprintf('Counterexample y_: [%g; %g]\n', y_(1), y_(2));
    fprintf('Bounds check: x_ in [%g,%g] x [%g,%g]: %d\n', ...
        x(1)-r(1), x(1)+r(1), x(2)-r(2), x(2)+r(2), ...
        all(x_ >= x - r) && all(x_ <= x + r));
    fprintf('A*y_ = %g, b = %g, violation: %d\n', A*y_, bunsafe, A*y_ <= bunsafe);
else
    fprintf('No counterexample found.\n');
end

fprintf('\n=== Conclusion ===\n');
if ~isempty(x_)
    fprintf('MATLAB verify produces counterexample: [%g; %g]\n', x_(1), x_(2));
    fprintf('This should match Python output.\n');
else
    fprintf('MATLAB verify found no counterexample.\n');
end
