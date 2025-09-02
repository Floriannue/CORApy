% Test MATLAB nnReshapeLayer behavior
idx_out = [1, 2; 3, 4];
input = [1; 2; 3; 4];

% Simulate what MATLAB does
idx_vec = idx_out(:);
result = input(idx_vec, :, :);

fprintf('idx_out:\n');
disp(idx_out);
fprintf('idx_vec:\n');
disp(idx_vec);
fprintf('input:\n');
disp(input);
fprintf('result:\n');
disp(result);
fprintf('result shape:\n');
disp(size(result));
