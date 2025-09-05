% Debug script to test Taylm empty creation in MATLAB
clear; clc;

% Test empty creation
n = 2;
empty_tay = taylm.empty(n);

fprintf('MATLAB Results:\n');
fprintf('Created empty Taylm with n=%d\n', n);
fprintf('Type: %s\n', class(empty_tay));
fprintf('Dim: %d\n', dim(empty_tay));
fprintf('Is empty: %d\n', isemptyobject(empty_tay));

% Check monomials structure
fprintf('Monomials type: %s\n', class(empty_tay.monomials));
fprintf('Monomials size: %s\n', mat2str(size(empty_tay.monomials)));

% Test with different dimensions
for n = [0, 1, 3, 5]
    empty_tay = taylm.empty(n);
    fprintf('n=%d: dim=%d, isempty=%d\n', n, dim(empty_tay), isemptyobject(empty_tay));
end
