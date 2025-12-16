% Test MATLAB permute behavior for sensitivity calculation
% S shape: (num_outputs, num_inputs, batch_size)
% Example: (5, 3, 2) - 5 outputs, 3 inputs, batch size 2

num_outputs = 5;
num_inputs = 3;
batch_size = 2;

% Create test S matrix
S = rand(num_outputs, num_inputs, batch_size);

fprintf('S shape: [%d, %d, %d]\n', size(S,1), size(S,2), size(S,3));

% MATLAB: sum(abs(S)) - sums over first dimension
sum_abs_S = sum(abs(S));
fprintf('sum(abs(S)) shape: [%d, %d]\n', size(sum_abs_S,1), size(sum_abs_S,2));
fprintf('sum(abs(S)):\n');
disp(sum_abs_S);

% MATLAB: permute(sum(abs(S)),[2 1 3])
sens_permuted = permute(sum_abs_S, [2 1 3]);
fprintf('permute(sum(abs(S)),[2 1 3]) shape: [%d, %d, %d]\n', size(sens_permuted,1), size(sens_permuted,2), size(sens_permuted,3));
fprintf('permute(sum(abs(S)),[2 1 3]):\n');
disp(sens_permuted);

% MATLAB: sens(:,:)
sens = sens_permuted(:,:);
fprintf('sens(:,:) final shape: [%d, %d]\n', size(sens,1), size(sens,2));
fprintf('sens:\n');
disp(sens);

% Test with xi and ri
xi = rand(num_inputs, batch_size);
ri = rand(num_inputs, batch_size);
fprintf('\nxi shape: [%d, %d]\n', size(xi,1), size(xi,2));
fprintf('ri shape: [%d, %d]\n', size(ri,1), size(ri,2));
fprintf('sens shape: [%d, %d]\n', size(sens,1), size(sens,2));

% MATLAB: zi = xi + ri.*sign(sens)
% Check if this works
try
    zi = xi + ri.*sign(sens);
    fprintf('SUCCESS: zi = xi + ri.*sign(sens) works\n');
    fprintf('zi shape: [%d, %d]\n', size(zi,1), size(zi,2));
catch ME
    fprintf('ERROR: %s\n', ME.message);
    % Try transpose
    fprintf('Trying with sens transposed:\n');
    zi = xi + ri.*sign(sens');
    fprintf('SUCCESS with transpose: zi = xi + ri.*sign(sens'') works\n');
    fprintf('zi shape: [%d, %d]\n', size(zi,1), size(zi,2));
end

