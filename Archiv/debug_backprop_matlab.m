% Debug script to understand MATLAB backpropZonotopeBatch behavior
% and generate expected output values for Python tests

% Open output file for writing
fid = fopen('matlab_output.txt', 'w');
if fid == -1
    error('Could not open output file');
end

% Create a simple LeakyReLU layer and wrap in neuralNetwork
alpha = 0.01;
layer = nnLeakyReLULayer(alpha);
nn = neuralNetwork({layer});

% Test inputs matching Python test
c = [1.0; 2.0];  % (2, 1) - will be reshaped to (2, 1, 1) internally
G = [0.1, 0.2; 0.3, 0.4];  % (2, 2) - will be reshaped to (2, 2, 1) internally

% Reshape to 3D for batch processing
c_3d = reshape(c, [2, 1, 1]);
G_3d = reshape(G, [2, 2, 1]);

% Forward pass options
options_forward = struct();
options_forward.nn = struct();
options_forward.nn.poly_method = 'bounds';
options_forward.nn.use_approx_error = false;
options_forward.nn.train = struct();
options_forward.nn.train.backprop = true;
options_forward.nn.train.exact_backprop = false;
options_forward.nn.train.num_init_gens = 2;
options_forward.nn.train.num_approx_err = 0;  % No approximation errors

% Prepare network for batch evaluation
nn.prepareForZonoBatchEval(c_3d, options_forward);

% Forward pass - call through neuralNetwork
[c_out, G_out] = nn.evaluateZonotopeBatch(c_3d, G_3d, options_forward);

fprintf('Forward pass results:\n');
fprintf('c_out shape: %s\n', mat2str(size(c_out)));
fprintf('G_out shape: %s\n', mat2str(size(G_out)));
fprintf('c_out:\n');
disp(c_out);
fprintf('G_out:\n');
disp(G_out);

% Check backprop storage (access through layer in network)
fprintf('\nBackprop storage after forward pass:\n');
fprintf('coeffs shape: %s\n', mat2str(size(nn.layers{1}.backprop.store.coeffs)));
fprintf('coeffs:\n');
disp(nn.layers{1}.backprop.store.coeffs);

% Get m (slope)
m = nn.layers{1}.backprop.store.coeffs;
fprintf('\nm (slope) shape: %s\n', mat2str(size(m)));
fprintf('m:\n');
disp(m);

% Test permute behavior
m_permuted = permute(m, [1, 3, 2]);
fprintf('\nm permuted [1 3 2] shape: %s\n', mat2str(size(m_permuted)));
fprintf('m_permuted:\n');
disp(m_permuted);

% Backprop inputs
gc = [0.5; 0.6];  % (2, 1) - will be reshaped to (2, 1, 1)
gG = [0.1, 0.2; 0.3, 0.4];  % (2, 2) - will be reshaped to (2, 2, 1)

% Reshape to 3D
gc_3d = reshape(gc, [2, 1, 1]);
gG_3d = reshape(gG, [2, 2, 1]);

% Backprop options - need full structure
options_backprop = struct();
options_backprop.nn = struct();
options_backprop.nn.poly_method = 'bounds';
options_backprop.nn.use_approx_error = false;
options_backprop.nn.train = struct();
options_backprop.nn.train.backprop = true;
options_backprop.nn.train.exact_backprop = false;
options_backprop.nn.train.num_approx_err = 0;

% Backprop pass - call through neuralNetwork
% Wrap in try-catch to continue even if it fails
try
    [gc_out, gG_out] = nn.backpropZonotopeBatch(c_3d, G_3d, gc_3d, gG_3d, options_backprop);
    backprop_success = true;
catch ME
    fprintf('Backprop failed: %s\n', ME.message);
    backprop_success = false;
    gc_out = [];
    gG_out = [];
end

fprintf('\nBackprop results:\n');
fprintf('gc_out shape: %s\n', mat2str(size(gc_out)));
fprintf('gG_out shape: %s\n', mat2str(size(gG_out)));
fprintf('gc_out:\n');
disp(gc_out);
fprintf('gG_out:\n');
disp(gG_out);

% Save results for Python comparison
save('backprop_test_results.mat', 'c_out', 'G_out', 'gc_out', 'gG_out', 'm', 'm_permuted');

% Print values for Python test
fprintf('\n=== Values for Python test ===\n');
fprintf('c_out shape: %s\n', mat2str(size(c_out)));
fprintf('c_out:\n');
fprintf('%s\n', mat2str(c_out));
fprintf('G_out shape: %s\n', mat2str(size(G_out)));
fprintf('G_out:\n');
fprintf('%s\n', mat2str(G_out));
fprintf('gc_out shape: %s\n', mat2str(size(gc_out)));
fprintf('gc_out:\n');
fprintf('%s\n', mat2str(gc_out));
fprintf('gG_out shape: %s\n', mat2str(size(gG_out)));
fprintf('gG_out:\n');
fprintf('%s\n', mat2str(gG_out));
fprintf('m shape: %s\n', mat2str(size(m)));
fprintf('m:\n');
fprintf('%s\n', mat2str(m));

% Also write to file (write what we have so far, even if backprop fails)
fprintf(fid, '=== MATLAB Output Values ===\n\n');
fprintf(fid, 'c_out shape: %s\n', mat2str(size(c_out)));
fprintf(fid, 'c_out:\n%s\n\n', mat2str(c_out));
fprintf(fid, 'G_out shape: %s\n', mat2str(size(G_out)));
fprintf(fid, 'G_out:\n%s\n\n', mat2str(G_out));
fprintf(fid, 'm (slope) shape: %s\n', mat2str(size(m)));
fprintf(fid, 'm:\n%s\n\n', mat2str(m));

% Write backprop results if available
if exist('backprop_success', 'var') && backprop_success
    fprintf(fid, 'gc_out shape: %s\n', mat2str(size(gc_out)));
    fprintf(fid, 'gc_out:\n%s\n\n', mat2str(gc_out));
    fprintf(fid, 'gG_out shape: %s\n', mat2str(size(gG_out)));
    fprintf(fid, 'gG_out:\n%s\n\n', mat2str(gG_out));
else
    fprintf(fid, 'Backprop failed - computing manually from forward pass\n');
    % Compute expected backprop manually: gc_out = gc .* m, gG_out = gG .* permute(m)
    % gc is (2,1,1), m is (2,2) - need to broadcast correctly
    % MATLAB: gc.*m where gc is (nk,1,bSz) and m is (nk,bSz)
    % Reshape m to (2,1,1) for first batch element
    m_batch1 = m(:,1);  % Extract first batch: (2,1)
    gc_manual = gc_3d .* reshape(m_batch1, [2, 1, 1]);  % (2,1,1) .* (2,1,1)
    
    % For gG: gG(:,genIds,:).*permute(m,[1 3 2])
    % m is (2,2), permute(m,[1 3 2]) gives (2,1,2)
    m_perm = permute(m, [1 3 2]);  % (2,1,2)
    % Use first batch slice
    gG_manual = gG_3d .* m_perm(:,:,1);  % (2,2,1) .* (2,1) broadcasts to (2,2,1)
    
    fprintf(fid, 'gc_out (manual) shape: %s\n', mat2str(size(gc_manual)));
    fprintf(fid, 'gc_out (manual):\n%s\n\n', mat2str(gc_manual));
    fprintf(fid, 'gG_out (manual) shape: %s\n', mat2str(size(gG_manual)));
    % Reshape to 2D for mat2str
    gG_2d = reshape(gG_manual, size(gG_manual, 1), []);
    fprintf(fid, 'gG_out (manual):\n%s\n\n', mat2str(gG_2d));
end

fclose(fid);
fprintf('Output written to matlab_output.txt\n');

