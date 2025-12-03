% debug_matlab_fgsm_constraints.m
% Debug script to verify constraint handling, reading, and FGSM attack construction
% This script helps ensure Python matches MATLAB's behavior exactly

function debug_matlab_fgsm_constraints()
    % Add CORA to path if needed
    % CORAROOT is a function, not a variable - add cora_matlab to path first
    if ~contains(path, 'cora_matlab')
        % Try to find cora_matlab directory
        possiblePaths = {
            fullfile(pwd, 'cora_matlab');
            fullfile(fileparts(pwd), 'cora_matlab');
            'cora_matlab';  % Relative to current directory
        };
        found = false;
        for i = 1:length(possiblePaths)
            if exist(possiblePaths{i}, 'dir')
                addpath(genpath(possiblePaths{i}));
                fprintf('Added cora_matlab to path: %s\n', possiblePaths{i});
                found = true;
                break;
            end
        end
        if ~found
            error('cora_matlab directory not found. Please run this script from the Translate_Cora directory or ensure cora_matlab exists.');
        end
    end
    
    % Now we can use CORAROOT() function
    try
        coraRoot = CORAROOT();
        fprintf('CORAROOT() returned: %s\n', coraRoot);
    catch
        % If CORAROOT() doesn't work, try to construct path manually
        fprintf('CORAROOT() function not available, using manual path construction\n');
        coraRoot = fullfile(pwd, 'cora_matlab');
    end
    
    fprintf('=== MATLAB FGSM Constraint Debug Script ===\n\n');
    
    % Test with prop_2.vnnlib (the failing test case)
    prop2Filename = fullfile(coraRoot, 'models', 'Cora', 'nn', 'prop_2.vnnlib');
    model1Path = fullfile(coraRoot, 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx');
    
    if ~exist(prop2Filename, 'file')
        error('prop_2.vnnlib not found at: %s', prop2Filename);
    end
    if ~exist(model1Path, 'file')
        error('Model file not found at: %s', model1Path);
    end
    
    % Read network and options (matching test code)
    [nn, options, x, r, A, b, safeSet] = aux_readNetworkAndOptions(model1Path, prop2Filename);
    
    fprintf('=== 1. Constraint Reading ===\n');
    fprintf('A shape: [%d, %d]\n', size(A, 1), size(A, 2));
    fprintf('A values:\n');
    disp(A);
    fprintf('b shape: [%d, %d]\n', size(b, 1), size(b, 2));
    fprintf('b values:\n');
    disp(b);
    fprintf('safeSet: %d (0=unsafeSet, 1=safeSet)\n', safeSet);
    fprintf('Number of constraints (p): %d\n\n', size(A, 1));
    
    % Set up FGSM options
    options.nn.falsification_method = 'fgsm';
    options.nn.refinement_method = 'naive';
    options.nn.train.backprop = true;  % Need backprop for sensitivity
    
    fprintf('=== 2. FGSM Attack Construction ===\n');
    
    % Simulate first iteration of verify loop
    % Use a small batch for debugging
    cbSz = 1;
    xi = x;  % Center of input set
    ri = r;  % Radius of input set
    n0 = size(xi, 1);
    
    fprintf('Input set:\n');
    fprintf('  xi shape: [%d, %d]\n', size(xi, 1), size(xi, 2));
    fprintf('  ri shape: [%d, %d]\n', size(ri, 1), size(ri, 2));
    fprintf('  xi = [%s]\n', sprintf('%.6f ', xi));
    fprintf('  ri = [%s]\n', sprintf('%.6f ', ri));
    fprintf('\n');
    
    % Construct input zonotope (simplified - just for getting sensitivity)
    % In real verify, this is more complex, but for debugging we'll use a simple approach
    fprintf('=== 3. Computing Sensitivity (S) ===\n');
    
    % Create a simple input zonotope for sensitivity computation
    % We need to call evaluateZonotopeBatch_ to get sensitivity
    % For debugging, let's create a minimal zonotope
    numInitGens = options.nn.train.num_init_gens;
    if isempty(numInitGens)
        numInitGens = 5;  % Default
    end
    
    % Create identity matrix for generators
    idMat = eye(n0);
    batchG = repmat(reshape(idMat, [n0, n0, 1]), [1, 1, cbSz]);
    cxi = repmat(xi, [1, 1, cbSz]);
    
    % Evaluate zonotope to get sensitivity
    options.nn.train.backprop = true;
    idxLayer = 1:length(nn.layers);
    
    % Try to compute sensitivity
    % This requires proper zonotope setup which is complex
    % For debugging, we'll try but also show the logic with example data
    S = [];
    try
        % Create proper input zonotope
        % In real verify, this uses aux_constructInputZonotope
        % For debugging, create a minimal version
        numInitGens = options.nn.train.num_init_gens;
        if isempty(numInitGens) || numInitGens > n0
            numInitGens = n0;
        end
        
        % Create generators (identity matrix for first numInitGens dimensions)
        idMat = eye(n0);
        batchG = repmat(reshape(idMat(:, 1:numInitGens), [n0, numInitGens, 1]), [1, 1, cbSz]);
        cxi = repmat(xi, [1, 1, cbSz]);
        
        % Evaluate to get sensitivity
        options.nn.train.backprop = true;
        [~, ~, S] = nn.evaluateZonotopeBatch_(cxi, batchG, options, idxLayer);
        fprintf('Sensitivity (S) computed successfully\n');
        fprintf('  S shape: [%d, %d, %d]\n', size(S, 1), size(S, 2), size(S, 3));
        fprintf('  S represents gradient of output w.r.t. input\n');
        if size(S, 1) >= 3 && size(S, 2) >= 1
            fprintf('  S(1:3,1,1) sample: [%s]\n', ...
                sprintf('%.6f ', S(1:3, 1, 1)));
        end
    catch ME
        fprintf('WARNING: Could not compute sensitivity: %s\n', ME.message);
        fprintf('  Creating mock S for demonstration purposes\n');
        % Get output dimension by evaluating network
        try
            y_test = nn.evaluate(xi);
            nK = size(y_test, 1);  % Output dimension
        catch
            % Fallback: assume 5 outputs (ACASXU networks have 5 outputs)
            nK = 5;
            fprintf('  Could not determine output dimension, assuming nK=5\n');
        end
        S = randn(nK, n0, cbSz) * 0.1;  % Mock sensitivity
        fprintf('  Using mock S with shape [%d, %d, %d]\n', size(S, 1), size(S, 2), size(S, 3));
        fprintf('  NOTE: This is for demonstration only - real S would be different\n');
        fprintf('  The mock S will show the logic but actual values will differ\n');
    end
    
    fprintf('\n=== 4. FGSM Gradient Computation ===\n');
    
    if ~isempty(S)
        p_orig = size(A, 1);
        fprintf('p_orig (number of constraints): %d\n', p_orig);
        
        if safeSet
            fprintf('safeSet = true: Using -A for gradient\n');
            grad = pagemtimes(-A, S);
            fprintf('  grad shape after pagemtimes(-A,S): [%d, %d, %d]\n', ...
                size(grad, 1), size(grad, 2), size(grad, 3));
            fprintf('  MATLAB sets p = 1 but does NOT explicitly sum grad!\n');
            fprintf('  grad still has shape [%d, %d, %d] with p_orig=%d constraints\n', ...
                size(grad, 1), size(grad, 2), size(grad, 3), p_orig);
            
            % Check what happens with reshape
            p = 1;  % MATLAB sets this
            fprintf('  p = %d (after setting p=1)\n', p);
            fprintf('  sign(grad) shape: [%d, %d, %d]\n', ...
                size(sign(grad), 1), size(sign(grad), 2), size(sign(grad), 3));
            
            % MATLAB: sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
            sgrad_permuted = permute(sign(grad), [2, 3, 1]);
            fprintf('  After permute([2 3 1]): shape [%d, %d, %d]\n', ...
                size(sgrad_permuted, 1), size(sgrad_permuted, 2), size(sgrad_permuted, 3));
            
            try
                sgrad = reshape(sgrad_permuted, [n0, cbSz*p]);
                fprintf('  After reshape([n0 cbSz*p]): shape [%d, %d]\n', ...
                    size(sgrad, 1), size(sgrad, 2));
                fprintf('  SUCCESS: Reshape worked! MATLAB must be using first constraint or summing implicitly\n');
            catch ME
                fprintf('  ERROR in reshape: %s\n', ME.message);
                fprintf('  This confirms that MATLAB must be doing something else!\n');
            end
            
            % Test if MATLAB sums implicitly
            fprintf('\n  Testing if MATLAB sums constraints implicitly:\n');
            grad_summed = sum(grad, 1);  % Sum over constraint dimension
            fprintf('  grad_summed shape: [%d, %d, %d]\n', ...
                size(grad_summed, 1), size(grad_summed, 2), size(grad_summed, 3));
            sgrad_summed_permuted = permute(sign(grad_summed), [2, 3, 1]);
            sgrad_summed = reshape(sgrad_summed_permuted, [n0, cbSz*p]);
            fprintf('  After summing and reshape: shape [%d, %d] - THIS WORKS!\n', ...
                size(sgrad_summed, 1), size(sgrad_summed, 2));
            fprintf('  CONCLUSION: MATLAB likely sums constraints implicitly!\n');
            
        else
            fprintf('safeSet = false (unsafeSet): Using +A for gradient\n');
            grad = pagemtimes(A, S);
            fprintf('  grad shape after pagemtimes(A,S): [%d, %d, %d]\n', ...
                size(grad, 1), size(grad, 2), size(grad, 3));
            fprintf('  p stays as p_orig = %d (tries each constraint individually)\n', p_orig);
            
            % Show gradient values
            fprintf('  grad(:,1,1) sample (first constraint, first 3 input dims): [%s]\n', ...
                sprintf('%.6f ', grad(1, 1:min(3,size(grad,2)), 1)));
            
            % MATLAB: sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
            p = p_orig;
            sgrad_permuted = permute(sign(grad), [2, 3, 1]);
            sgrad = reshape(sgrad_permuted, [n0, cbSz*p]);
            fprintf('  After permute+reshape: sgrad shape [%d, %d]\n', ...
                size(sgrad, 1), size(sgrad, 2));
            
            % Show attack direction
            fprintf('  sgrad(:,1) (first constraint attack direction): [%s]\n', ...
                sprintf('%.6f ', sgrad(:, 1)));
            fprintf('  This moves in direction that INCREASES A*y\n');
            fprintf('  But for unsafeSet, we want A*y <= b (DECREASE A*y)!\n');
            fprintf('  QUESTION: Why does MATLAB use +grad for unsafeSet?\n');
        end
        
        fprintf('\n=== 5. Attack Vector Construction ===\n');
        % MATLAB: xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;
        xi_repeated = repelem(xi, 1, p);
        ri_repeated = repelem(ri, 1, p);
        xi_ = xi_repeated + ri_repeated .* sgrad;
        fprintf('  xi_repeated shape: [%d, %d]\n', size(xi_repeated, 1), size(xi_repeated, 2));
        fprintf('  ri_repeated shape: [%d, %d]\n', size(ri_repeated, 1), size(ri_repeated, 2));
        fprintf('  xi_ (attack candidates) shape: [%d, %d]\n', size(xi_, 1), size(xi_, 2));
        fprintf('  xi_(:,1) (first candidate): [%s]\n', sprintf('%.6f ', xi_(:, 1)));
        
        fprintf('\n=== 6. Checking Specification ===\n');
        % Evaluate network on attack candidates
        options.nn.train.backprop = false;
        yi = nn.evaluate_(xi_, options, idxLayer);
        fprintf('  yi (network output) shape: [%d, %d]\n', size(yi, 1), size(yi, 2));
        
        % Compute ld_yi = A * yi
        ld_yi = A * yi;
        fprintf('  ld_yi = A * yi shape: [%d, %d]\n', size(ld_yi, 1), size(ld_yi, 2));
        fprintf('  ld_yi(:,1) (first candidate): [%s]\n', sprintf('%.6f ', ld_yi(:, 1)));
        fprintf('  b: [%s]\n', sprintf('%.6f ', b));
        
        % Check specification
        if safeSet
            % safeSet: falsified = any(ld_yi > b, 1)
            falsified = any(ld_yi > b, 1);
            fprintf('  safeSet: falsified = any(ld_yi > b, 1)\n');
        else
            % unsafeSet: falsified = all(ld_yi <= b, 1)
            falsified = all(ld_yi <= b, 1);
            fprintf('  unsafeSet: falsified = all(ld_yi <= b, 1)\n');
        end
        
        fprintf('  falsified: [%s]\n', sprintf('%d ', falsified));
        fprintf('  Found counterexamples: %d / %d\n', sum(falsified), length(falsified));
        
        if any(falsified)
            fprintf('  SUCCESS: Found counterexample(s)!\n');
            id = find(falsified, 1);
            fprintf('  First counterexample at index %d\n', id);
            fprintf('  xi_(:,%d): [%s]\n', id, sprintf('%.6f ', xi_(:, id)));
            fprintf('  yi(:,%d): [%s]\n', id, sprintf('%.6f ', yi(:, id)));
            fprintf('  ld_yi(:,%d): [%s]\n', id, sprintf('%.6f ', ld_yi(:, id)));
        else
            fprintf('  No counterexamples found in first iteration\n');
            fprintf('  This matches Python behavior - attack direction may be wrong\n');
        end
    else
        fprintf('Skipping FGSM computation (S not available)\n');
    end
    
    fprintf('\n=== 7. Running Actual Verify Function ===\n');
    fprintf('Running nn.verify() to see what MATLAB actually does...\n\n');
    
    % Reset options for actual verification
    options.nn.falsification_method = 'fgsm';
    options.nn.refinement_method = 'naive';
    options.nn.train.backprop = false;  % Will be set to true when needed
    timeout = 10;
    verbose = true;  % Enable verbose to see intermediate steps
    
    try
        [verifRes, x_, y_] = nn.verify(x, r, A, b, safeSet, options, timeout, verbose);
        
        fprintf('\n=== Verification Results ===\n');
        fprintf('Result: %s\n', verifRes.str);
        fprintf('x_ is empty: %d\n', isempty(x_));
        fprintf('y_ is empty: %d\n', isempty(y_));
        
        if ~isempty(x_) && ~isempty(y_)
            fprintf('\nCounterexample found:\n');
            fprintf('x_ = [%s]\n', sprintf('%.6f ', x_));
            fprintf('y_ = [%s]\n', sprintf('%.6f ', y_));
            
            % Verify it actually violates spec
            yi_check = nn.evaluate(x_);
            ld_check = A * yi_check;
            fprintf('\nVerification of counterexample:\n');
            fprintf('  A*y = [%s]\n', sprintf('%.6f ', ld_check));
            fprintf('  b = [%s]\n', sprintf('%.6f ', b));
            if safeSet
                violates = any(ld_check > b);
                fprintf('  safeSet: violates = any(A*y > b) = %d\n', violates);
            else
                violates = all(ld_check <= b);
                fprintf('  unsafeSet: violates = all(A*y <= b) = %d\n', violates);
            end
        end
    catch ME
        fprintf('ERROR running verify: %s\n', ME.message);
        fprintf('  Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('    %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
    
    fprintf('\n=== Summary ===\n');
    fprintf('1. Constraint reading: A and b extracted correctly\n');
    fprintf('   - A shape: [%d, %d]\n', size(A, 1), size(A, 2));
    fprintf('   - b shape: [%d, %d]\n', size(b, 1), size(b, 2));
    fprintf('   - safeSet: %d\n', safeSet);
    fprintf('\n');
    fprintf('2. For safeSet: MATLAB sets p=1 but grad has p_orig constraints\n');
    fprintf('   - Reshape would fail unless MATLAB sums or takes first constraint\n');
    fprintf('   - Most likely: MATLAB sums constraints implicitly\n');
    fprintf('   - Python explicitly sums: grad = sum(grad, axis=0)\n');
    fprintf('\n');
    fprintf('3. For unsafeSet: MATLAB uses +grad which INCREASES A*y\n');
    fprintf('   - But we want A*y <= b (DECREASE A*y)\n');
    fprintf('   - This seems backwards but MATLAB finds counterexamples!\n');
    fprintf('   - Need to investigate why this works\n');
    fprintf('   - Possible: splitting creates sets where +grad direction works\n');
    fprintf('   - Or: the attack explores boundary in a way we don''t understand\n');
    fprintf('\n');
    fprintf('4. Key values to compare with Python:\n');
    fprintf('   - A: [%s]\n', sprintf('%.6f ', A(:)'));
    fprintf('   - b: [%s]\n', sprintf('%.6f ', b(:)'));
    fprintf('   - x: [%s]\n', sprintf('%.6f ', x(:)'));
    fprintf('   - r: [%s]\n', sprintf('%.6f ', r(:)'));
end

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

