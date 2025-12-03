% Debug script to compare r, ri_, and m values with Python
% This uses the REAL verify() function instead of simplified helper calls

function debug_matlab_generator_collapse()
    % Auto-detect CORAROOT
    scriptPath = mfilename('fullpath');
    [scriptDir, ~, ~] = fileparts(scriptPath);
    workspaceRoot = scriptDir;
    
    % Look for cora_matlab directory
    coraMatlabPath = fullfile(workspaceRoot, 'cora_matlab');
    if ~exist(coraMatlabPath, 'dir')
        error('cora_matlab directory not found. Please ensure it exists in the workspace root.');
    end
    
    % Add CORA to path
    addpath(genpath(coraMatlabPath));
    
    % Set CORAROOT if not already set
    if isempty(getenv('CORAROOT'))
        CORAROOT();
    end
    
    fprintf('CORAROOT() returned: %s\n\n', CORAROOT());
    fprintf('=== MATLAB Generator Collapse Debug Script ===\n');
    fprintf('Using REAL verify() function\n\n');
    
    % Load network and specification
    modelPath = fullfile(coraMatlabPath, 'models', 'Cora', 'nn', 'ACASXU_run2a_1_2_batch_2000.onnx');
    if ~exist(modelPath, 'file')
        error('Model file not found at: %s', modelPath);
    end
    
    specPath = fullfile(coraMatlabPath, 'models', 'Cora', 'nn', 'prop_1.vnnlib');
    if ~exist(specPath, 'file')
        error('Specification file not found at: %s', specPath);
    end
    
    fprintf('Loading network and specification...\n');
    % Use 'BSSC' format to match Python test
    nn = neuralNetwork.readONNXNetwork(modelPath, false, 'BSSC');
    
    % Read VNNLIB specification (matches Python test)
    [X0, specs] = vnnlib2cora(specPath);
    
    % Extract x, r, A, b, safeSet from specs (matches Python test)
    x = 1/2*(X0{1}.sup + X0{1}.inf);  % Center of input interval
    r = 1/2*(X0{1}.sup - X0{1}.inf);  % Radius
    
    % Extract specification
    if isa(specs.set, 'halfspace')
        A = specs.set.c';
        b = specs.set.d;
        safeSet = true;
    elseif isa(specs.set, 'polytope')
        A = specs.set.A;
        b = specs.set.b;
        safeSet = strcmp(specs.type, 'safeSet');
    else
        error('Unsupported specification type: %s', class(specs.set));
    end
    
    fprintf('Network loaded: %d layers\n', length(nn.layers));
    fprintf('Input dimension: %d\n', size(x, 1));
    fprintf('Specification: A shape [%d, %d], b shape [%d, %d], safeSet=%d\n\n', ...
        size(A, 1), size(A, 2), size(b, 1), size(b, 2));
    
    % Set options to match Python test
    options.nn = struct(...
        'use_approx_error', false, ...
        'poly_method', 'bounds', ...
        'train', struct(...
            'backprop', false, ...
            'mini_batch_size', 32, ...
            'num_init_gens', 5, ...
            'num_approx_err', 0 ...
        ) ...
    );
    
    % Set additional options
    options.nn.refinement_method = 'naive';
    options.nn.falsification_method = 'fgsm';
    options.nn.interval_center = false;
    
    % Validate options (this fills in defaults)
    options = nnHelper.validateNNoptions(options, true);
    
    % Override after validation
    options.nn.interval_center = false;
    options.nn.use_approx_error = false;
    options.nn.train.num_init_gens = 5;
    options.nn.train.num_approx_err = 0;
    
    fprintf('==== Starting Verification (using REAL verify() function) ===\n');
    fprintf('Note: This uses the actual verify() implementation\n');
    fprintf('Debug logging can be added to verify.m helper functions if needed\n\n');
    fprintf('============================================================\n\n');
    
    % Use the REAL verify() function
    % Syntax: [res, x_, y_] = verify(nn, x, r, A, b, safeSet, options, timeout, verbose)
    timeout = 10.0;
    verbose = true;  % Enable verbose output to see iteration stats
    
    % Call the real verify function - this uses all the real implementations
    [verifRes, x_, y_] = nn.verify(x, r, A, b, safeSet, options, timeout, verbose);
    
    fprintf('\n============================================================\n');
    fprintf('=== Verification Result: %s ===\n', verifRes.str);
    fprintf('=== Debug Complete ===\n');
end
