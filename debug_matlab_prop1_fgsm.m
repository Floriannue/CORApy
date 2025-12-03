% Debug script to trace MATLAB's verification behavior for prop_1 with FGSM
% This matches the failing Python test case

% Add CORA to path
addpath(genpath('cora_matlab'));

% Specify the model path
model1Path = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
prop1Filename = [CORAROOT '/models/Cora/nn/prop_1.vnnlib'];

% Read network and options (matching test)
nn = neuralNetwork.readONNXNetwork(model1Path,false,'BSSC');
[X0,specs] = vnnlib2cora(prop1Filename);

% Extract input set
x = 1/2*(X0{1}.sup + X0{1}.inf);
r = 1/2*(X0{1}.sup - X0{1}.inf);

% Extract specification
if isa(specs.set,'halfspace')
    A = specs.set.c';
    b = specs.set.d;
else
    A = specs.set.A;
    b = specs.set.b;
end
safeSet = strcmp(specs.type,'safeSet');

fprintf('=== Test Setup ===\n');
fprintf('x = [%s]\n', num2str(x'));
fprintf('r = [%s]\n', num2str(r'));
fprintf('A = [%s]\n', num2str(A));
fprintf('b = %g\n', b);
fprintf('safeSet = %d\n', safeSet);

% Create evaluation options (matching test)
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds',...
    'train',struct(...
        'backprop',false,...
        'mini_batch_size',2^8 ...
    ) ...
);
options = nnHelper.validateNNoptions(options,true);
options.nn.interval_center = false;

% Test 'naive'-splitting and 'fgsm'-falsification
options.nn.falsification_method = 'fgsm';
options.nn.refinement_method = 'naive';

% Set timeout
timeout = 10;
verbose = false;  % Set to false to reduce output

% Do verification
fprintf('\n=== Running Verification ===\n');
[verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

fprintf('\n=== Results ===\n');
fprintf('Result: %s\n', verifRes.str);
fprintf('x_ is empty: %d\n', isempty(x_));
fprintf('y_ is empty: %d\n', isempty(y_));

if ~isempty(x_)
    fprintf('\nCounterexample found:\n');
    fprintf('x_ = [%s]\n', num2str(x_'));
    fprintf('y_ = [%s]\n', num2str(y_'));
    
    % Re-evaluate to check
    yi_check = nn.evaluate(x_);
    ld_check = A * yi_check;
    fprintf('A*y = %g, b = %g\n', ld_check, b);
    fprintf('A*y <= b: %d\n', ld_check <= b);
    fprintf('Violates spec (all(A*y <= b)): %d\n', all(ld_check <= b));
end

% Expected: MATLAB should return VERIFIED (or UNKNOWN), not COUNTEREXAMPLE
% Python is finding a counterexample that MATLAB doesn't find
fprintf('\n=== Expected ===\n');
fprintf('MATLAB test expects: NOT COUNTEREXAMPLE (i.e., VERIFIED or UNKNOWN)\n');
fprintf('Actual result: %s\n', verifRes.str);

if strcmp(verifRes.str, 'COUNTEREXAMPLE')
    fprintf('\nWARNING: MATLAB found COUNTEREXAMPLE, but test expects it not to!\n');
    fprintf('This suggests the test or MATLAB behavior changed.\n');
else
    fprintf('\nSUCCESS: MATLAB matches expected behavior (not COUNTEREXAMPLE)\n');
end

