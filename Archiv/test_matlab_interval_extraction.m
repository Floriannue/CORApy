% test_matlab_interval_extraction - Test how MATLAB extracts values from Interval matrices

% Create a simple Interval matrix
I = interval([-0.1, -0.05; -0.05, -0.1], [0.1, 0.05; 0.05, 0.1]);

% Test 1: Direct indexing
fprintf('Test 1: Direct indexing\n');
val1 = I(1,1);
fprintf('  I(1,1) type: %s\n', class(val1));
if isa(val1, 'interval')
    fprintf('  I(1,1) is Interval: [%.6f, %.6f]\n', val1.inf, val1.sup);
end

% Test 2: Diagonal extraction
fprintf('\nTest 2: Diagonal extraction\n');
diag_vals = diag(I);
fprintf('  diag(I) type: %s\n', class(diag_vals));
fprintf('  diag(I) size: %s\n', mat2str(size(diag_vals)));
if isa(diag_vals, 'interval')
    fprintf('  diag(I) is Interval matrix\n');
    fprintf('  diag(I)(1) = [%.6f, %.6f]\n', diag_vals(1).inf, diag_vals(1).sup);
    if length(diag_vals) > 1
        fprintf('  diag(I)(2) = [%.6f, %.6f]\n', diag_vals(2).inf, diag_vals(2).sup);
    end
end

% Test 3: Assignment to numeric array
fprintf('\nTest 3: Assignment to numeric array\n');
G = zeros(2, 1);
try
    G(1:2) = 0.5 * diag(I);
    fprintf('  Assignment successful\n');
    fprintf('  G type: %s\n', class(G));
    fprintf('  G values: %s\n', mat2str(G));
catch ME
    fprintf('  Assignment failed: %s\n', ME.message);
end

% Test 4: Using supremum/infimum
fprintf('\nTest 4: Using supremum/infimum\n');
I_sup = supremum(I);
I_inf = infimum(I);
I_center = center(I);
fprintf('  supremum(I) diagonal: %s\n', mat2str(diag(I_sup)));
fprintf('  infimum(I) diagonal: %s\n', mat2str(diag(I_inf)));
fprintf('  center(I) diagonal: %s\n', mat2str(diag(I_center)));

% Test 5: Matrix multiplication result
fprintf('\nTest 5: Matrix multiplication with Interval\n');
Zmat = [1, 1, 0.5; 0, 0, 0.3];
Q = interval([-0.1, -0.05; -0.05, -0.1], [0.1, 0.05; 0.05, 0.1]);
quadMat = Zmat' * Q * Zmat;
fprintf('  quadMat type: %s\n', class(quadMat));
if isa(quadMat, 'interval')
    fprintf('  quadMat is Interval matrix\n');
    fprintf('  quadMat(1,1) = [%.6f, %.6f]\n', quadMat(1,1).inf, quadMat(1,1).sup);
    fprintf('  diag(quadMat(2:3,2:3)):\n');
    diag_sub = diag(quadMat(2:3, 2:3));
    for i = 1:length(diag_sub)
        fprintf('    [%.6f, %.6f]\n', diag_sub(i).inf, diag_sub(i).sup);
    end
    % Try assignment
    G_test = zeros(2, 1);
    try
        G_test(1:2) = 0.5 * diag(quadMat(2:3, 2:3));
        fprintf('  Assignment to G_test successful\n');
        fprintf('  G_test values: %s\n', mat2str(G_test));
    catch ME
        fprintf('  Assignment failed: %s\n', ME.message);
    end
end
