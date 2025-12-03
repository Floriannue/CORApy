% Detailed debug of zonotack attack to find why Python gets out-of-bounds
% This will trace the exact computation step by step

% Add CORA to path
addpath(genpath([CORAROOT '/cora_matlab']));

fprintf('=== Detailed Zonotack Attack Debug ===\n\n');

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
verbose = false;
timeout = 2;

% Options
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds',...
    'train',struct(...
        'backprop',false,...
        'mini_batch_size',512,...
        'num_init_gens',2 ...
    ) ...
);
options = nnHelper.validateNNoptions(options,true);
options.nn.interval_center = false;
options.nn.falsification_method = 'zonotack';
options.nn.refinement_method = 'zonotack';

% Get numInitGens
n0 = size(x,1);
numInitGens = min(options.nn.train.num_init_gens, n0);

fprintf('Configuration:\n');
fprintf('  x: [%g; %g]\n', x(1), x(2));
fprintf('  r: [%g; %g]\n', r(1), r(2));
fprintf('  numInitGens: %d\n', numInitGens);
fprintf('  A: [%g %g]\n', A(1), A(2));
fprintf('  b: %g\n', bunsafe);
fprintf('\n');

% Run verification and capture intermediate values
% We need to modify verify.m to print intermediate values, or use a simpler approach
% Let's just run it and see what counterexample we get
[res,x_,y_] = nn.verify(x,r,A,bunsafe,safeSet,options,timeout,verbose);

fprintf('Result: %s\n', res.str);
if ~isempty(x_)
    fprintf('Counterexample x_: [%g; %g]\n', x_(1), x_(2));
    fprintf('Counterexample y_: [%g; %g]\n', y_(1), y_(2));
    
    % Verify it's in bounds
    in_bounds = all(x_ >= x - r) && all(x_ <= x + r);
    fprintf('x_ in bounds [%g,%g] x [%g,%g]: %d\n', ...
        x(1)-r(1), x(1)+r(1), x(2)-r(2), x(2)+r(2), in_bounds);
    
    % Verify specification violation
    yi = nn.evaluate(x_);
    ld_yi = A * yi;
    violates = (ld_yi <= bunsafe);
    fprintf('A*y_ = %g, b = %g, violates (A*y_ <= b): %d\n', ld_yi, bunsafe, violates);
end

fprintf('\n=== Python found ===\n');
fprintf('zi = [-1; -2]\n');
fprintf('yi = [1.86673729; -2.67599051]\n');
fprintf('A*yi = %g\n', A * [1.86673729; -2.67599051]);
fprintf('zi in bounds: %d (should be 0, -2 is out of bounds)\n', ...
    all([-1; -2] >= x - r) && all([-1; -2] <= x + r));

fprintf('\n=== Analysis ===\n');
fprintf('If Python finds out-of-bounds counterexample, there is a bug.\n');
fprintf('The zonotack attack should stay within [xi-ri, xi+ri] by construction.\n');


