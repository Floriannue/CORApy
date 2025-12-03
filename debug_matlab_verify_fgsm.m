% Debug script to compare MATLAB verify behavior with Python
% This script runs the same test case as testnn_neuralNetwork_verify

% Add CORA to path
addpath(genpath([CORAROOT '/cora_matlab']));

% First test case: prop_1.vnnlib
model1Path = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
prop1Filename = [CORAROOT '/models/Cora/nn/prop_1.vnnlib'];

% Set a timeout of 10s
timeout = 10;
verbose = true;

% Read network and options
[nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model1Path,prop1Filename);

% Test 'naive'-splitting and 'fgsm'-falsification
options.nn.falsification_method = 'fgsm';
options.nn.refinement_method = 'naive';

fprintf('=== MATLAB Verify Debug ===\n');
fprintf('Model: %s\n', model1Path);
fprintf('Spec: %s\n', prop1Filename);
fprintf('safeSet: %d\n', safeSet);
fprintf('falsification_method: %s\n', options.nn.falsification_method);
fprintf('refinement_method: %s\n', options.nn.refinement_method);
fprintf('x shape: [%d, %d]\n', size(x,1), size(x,2));
fprintf('r shape: [%d, %d]\n', size(r,1), size(r,2));
fprintf('A shape: [%d, %d]\n', size(A,1), size(A,2));
fprintf('b shape: [%d, %d]\n', size(b,1), size(b,2));
fprintf('b value: %s\n', mat2str(b));

% Do verification
[verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

fprintf('\n=== Results ===\n');
fprintf('Result: %s\n', verifRes.str);
fprintf('x_ empty: %d\n', isempty(x_));
fprintf('y_ empty: %d\n', isempty(y_));

if ~isempty(x_)
    fprintf('x_ shape: [%d, %d]\n', size(x_,1), size(x_,2));
    fprintf('x_ value: %s\n', mat2str(x_));
end
if ~isempty(y_)
    fprintf('y_ shape: [%d, %d]\n', size(y_,1), size(y_,2));
    fprintf('y_ value: %s\n', mat2str(y_));
end

% Check assertion
assert(~strcmp(verifRes.str,'COUNTEREXAMPLE') & isempty(x_) & isempty(y_));
fprintf('\n=== Assertion passed ===\n');

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

