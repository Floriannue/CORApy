% Debug script to compare MATLAB verify falsification logic
% This matches debug_verify_falsification.py
% 
% This script runs the same test case as the Python debug script and shows
% detailed debug output from verify.m (which has DEBUG prints added to
% aux_checkPoints and FGSM sections)

% Load the same test case (using default CORAROOT from OneDrive)
model1Path = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
prop1Filename = [CORAROOT '/models/Cora/nn/prop_1.vnnlib'];

if exist(model1Path, 'file') && exist(prop1Filename, 'file')
    [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model1Path,prop1Filename);
    
    % Set options to match Python test (same as testnn_neuralNetwork_verify.m)
    options.nn.falsification_method = 'fgsm';
    options.nn.refinement_method = 'naive';
    
    % Re-validate options to ensure all defaults are set
    % This is what verify.m does at line 64
    options = nnHelper.validateNNoptions(options, true);
    
    % Ensure required fields exist (validateNNoptions should set these, but be safe)
    if ~isfield(options.nn, 'num_splits')
        options.nn.num_splits = 5;  % Default
    end
    if ~isfield(options.nn, 'num_dimensions')
        options.nn.num_dimensions = 1;  % Default
    end
    if ~isfield(options.nn, 'num_neuron_splits')
        options.nn.num_neuron_splits = 0;  % Default
    end
    if ~isfield(options.nn, 'num_relu_constraints')
        options.nn.num_relu_constraints = 0;  % Default
    end
    
    fprintf('=== MATLAB VERIFY DEBUG ===\n');
    fprintf('Input x shape: [%d %d], r shape: [%d %d]\n', size(x,1), size(x,2), size(r,1), size(r,2));
    fprintf('A shape: [%d %d], b shape: [%d %d], safeSet: %d\n', size(A,1), size(A,2), size(b,1), size(b,2));
    fprintf('b value: [%f]\n', b);
    fprintf('b type: %s, b ndims: %d\n', class(b), ndims(b));
    fprintf('\n=== RUNNING VERIFY (look for DEBUG output from aux_checkPoints and FGSM) ===\n\n');
    
    % Verify we're using the correct verify.m file (should be in OneDrive now)
    verifyPath = fullfile(CORAROOT, 'nn', '@neuralNetwork', 'verify.m');
    if exist(verifyPath, 'file')
        fileInfo = dir(verifyPath);
        fprintf('Using verify.m from: %s\n', verifyPath);
        fprintf('File last modified: %s\n', fileInfo.date);
    else
        fprintf('ERROR: verify.m not found at: %s\n', verifyPath);
    end
    
    % Clear function cache to force MATLAB to reload the modified file
    clear('neuralNetwork');
    clear('functions');
    rehash path;
    
    fprintf('Cleared function cache. MATLAB will reload verify.m\n\n');
    
    % Run verification with EXACT same settings as test
    timeout = 10;  % Same as test (line 41 in testnn_neuralNetwork_verify.m)
    verbose = true;  % Same as test (line 32 in testnn_neuralNetwork_verify.m)
    [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
    
    % Check if verifRes is a struct or string
    if isstruct(verifRes) && isfield(verifRes, 'str')
        resultStr = verifRes.str;
    else
        resultStr = verifRes;
    end
    fprintf('\n=== VERIFICATION RESULT ===\n');
    fprintf('Result: %s\n', resultStr);
    fprintf('x_ is empty: %d, y_ is empty: %d\n', isempty(x_), isempty(y_));
    if ~isempty(x_)
        fprintf('x_ shape: [%d %d], x_ value: [%f %f %f %f %f]\n', size(x_,1), size(x_,2), x_);
    end
    if ~isempty(y_)
        fprintf('y_ shape: [%d %d], y_ value: [%f %f %f %f %f]\n', size(y_,1), size(y_,2), y_);
    end
    
    % Check if counterexample is valid
    if ~isempty(x_) && ~isempty(y_)
        % Check if x_ is within input bounds
        x_lower = x - r;
        x_upper = x + r;
        fprintf('\nInput bounds check:\n');
        fprintf('x (center): [%f %f %f %f %f]\n', x);
        fprintf('r (radius): [%f %f %f %f %f]\n', r);
        fprintf('x_lower: [%f %f %f %f %f]\n', x_lower);
        fprintf('x_upper: [%f %f %f %f %f]\n', x_upper);
        fprintf('x_ (counterexample): [%f %f %f %f %f]\n', x_);
        in_bounds = all(x_ >= x_lower) && all(x_ <= x_upper);
        fprintf('x_ within bounds: %d\n', in_bounds);
        if ~in_bounds
            fprintf('OUT OF BOUNDS! Violations:\n');
            fprintf('  Below lower: [%d %d %d %d %d]\n', x_ < x_lower);
            fprintf('  Above upper: [%d %d %d %d %d]\n', x_ > x_upper);
        end
        
        yi = nn.evaluate(x_);
        ld_yi = A*yi;
        fprintf('\nCounterexample check:\n');
        fprintf('yi from evaluate: [%f %f %f %f %f]\n', yi);
        fprintf('y_ from verify: [%f %f %f %f %f]\n', y_);
        fprintf('Difference: [%f %f %f %f %f]\n', abs(y_ - yi));
        fprintf('A*yi: [%f]\n', ld_yi);
        fprintf('b: [%f]\n', b);
        fprintf('A*yi - b: [%f]\n', ld_yi - b);
        fprintf('ld_yi shape: [%d %d], b shape: [%d %d]\n', size(ld_yi,1), size(ld_yi,2), size(b,1), size(b,2));
        fprintf('ld_yi <= b: [%d]\n', ld_yi <= b);
        if safeSet
            violates = any(ld_yi > b);
            fprintf('Violates (safeSet): %d (any(A*y > b))\n', violates);
        else
            violates = all(ld_yi <= b);
            fprintf('Violates (unsafeSet): %d (all(A*y <= b))\n', violates);
            fprintf('all(ld_yi <= b, 1): [%d]\n', all(ld_yi <= b, 1));
        end
    end
else
    fprintf('Files not found, skipping debug\n');
end

% Auxiliary function (from testnn_neuralNetwork_verify.m)
function [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,vnnlibPath)
  % Create evaluation options.
  options.nn = struct(...
      'use_approx_error',true,...
      'poly_method','bounds',...
      'train',struct(...
          'backprop',false,...
          'mini_batch_size',2^8 ...
      ) ...
  );
  % Set default training parameters
  options = nnHelper.validateNNoptions(options,true);
  options.nn.interval_center = false;

  % Read the neural network.
  nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');

  % Read the input set and specification.
  [X0,specs] = vnnlib2cora(vnnlibPath);

  % Extract input set.
  x = 1/2*(X0{1}.sup + X0{1}.inf);
  r = 1/2*(X0{1}.sup - X0{1}.inf);
  
  % Extract specification.
  if isa(specs.set,'halfspace')
      A = specs.set.c';
      b = specs.set.d;
  else
      A = specs.set.A;
      b = specs.set.b;
  end
  safeSet = strcmp(specs.type,'safeSet');
end

