% test_matlab_quadmap_interval - Test how MATLAB's quadMap handles Interval Hessian

% Create a simple zonotope (matching Python test)
Z = zonotope([1, 1, 0.5; 0, 0, 0.3]);

% Create an Interval Hessian (2x2 interval matrix)
% This simulates what happens when H[i] is an Interval
H_interval = interval([-0.1, -0.05; -0.05, -0.1], [0.1, 0.05; 0.05, 0.1]);

H = {H_interval, H_interval};  % Two dimensions

fprintf('Testing MATLAB quadMap with Interval Hessian:\n');
fprintf('Z center: %s\n', mat2str(center(Z)));
fprintf('Z generators shape: %s\n', mat2str(size(generators(Z))));
fprintf('H{1} type: %s\n', class(H{1}));

% Test quadMap
try
    errorSec = 0.5 * quadMap(Z, H);
    fprintf('\nquadMap result:\n');
    fprintf('  Center: %s\n', mat2str(center(errorSec)));
    fprintf('  Generators shape: %s\n', mat2str(size(generators(errorSec))));
    fprintf('  Radius: %s\n', mat2str(sum(abs(generators(errorSec)), 2)));
    fprintf('  Radius max: %.6e\n', max(sum(abs(generators(errorSec)), 2)));
catch ME
    fprintf('ERROR in quadMap: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end

% Test the matrix multiplication step manually
fprintf('\nTesting matrix multiplication manually:\n');
Zmat = [center(Z), generators(Z)];
fprintf('Zmat shape: %s\n', mat2str(size(Zmat)));

% Test: Zmat'*H{1}*Zmat
try
    quadMat = Zmat' * H{1} * Zmat;
    fprintf('quadMat type: %s\n', class(quadMat));
    
    if isa(quadMat, 'interval')
        fprintf('quadMat is Interval matrix\n');
        fprintf('  Size: %s\n', mat2str(size(quadMat)));
        fprintf('  quadMat(1,1) = [%.6f, %.6f]\n', quadMat(1,1).inf, quadMat(1,1).sup);
        
        % Test extraction methods
        fprintf('\nTesting extraction methods:\n');
        
        % Method 1: supremum
        quadMat_sup = supremum(quadMat);
        fprintf('  supremum(quadMat) diagonal: %s\n', mat2str(diag(quadMat_sup(2:3, 2:3))));
        
        % Method 2: infimum
        quadMat_inf = infimum(quadMat);
        fprintf('  infimum(quadMat) diagonal: %s\n', mat2str(diag(quadMat_inf(2:3, 2:3))));
        
        % Method 3: center
        quadMat_center = center(quadMat);
        fprintf('  center(quadMat) diagonal: %s\n', mat2str(diag(quadMat_center(2:3, 2:3))));
        
        % Method 4: Try to extract directly
        fprintf('\nTrying direct extraction:\n');
        try
            diag_vals = diag(quadMat(2:3, 2:3));
            fprintf('  diag(quadMat(2:3,2:3)) type: %s\n', class(diag_vals));
            if isa(diag_vals, 'interval')
                fprintf('  diag_vals is Interval\n');
                fprintf('  diag_vals(1) = [%.6f, %.6f]\n', diag_vals(1).inf, diag_vals(1).sup);
            end
        catch ME2
            fprintf('  Direct extraction failed: %s\n', ME2.message);
        end
        
        % Method 5: Try assignment
        fprintf('\nTrying assignment:\n');
        G_test = zeros(2, 1);
        try
            % Try with supremum
            G_test(1:2) = 0.5 * diag(supremum(quadMat(2:3, 2:3)));
            fprintf('  Assignment with supremum: G_test = %s\n', mat2str(G_test));
        catch ME3
            fprintf('  Assignment with supremum failed: %s\n', ME3.message);
        end
        
        try
            % Try with infimum
            G_test(1:2) = 0.5 * diag(infimum(quadMat(2:3, 2:3)));
            fprintf('  Assignment with infimum: G_test = %s\n', mat2str(G_test));
        catch ME4
            fprintf('  Assignment with infimum failed: %s\n', ME4.message);
        end
        
        try
            % Try with center
            G_test(1:2) = 0.5 * diag(center(quadMat(2:3, 2:3)));
            fprintf('  Assignment with center: G_test = %s\n', mat2str(G_test));
        catch ME5
            fprintf('  Assignment with center failed: %s\n', ME5.message);
        end
        
    else
        fprintf('quadMat is NOT an Interval (type: %s)\n', class(quadMat));
        fprintf('  quadMat diagonal: %s\n', mat2str(diag(quadMat(2:3, 2:3))));
    end
catch ME
    fprintf('ERROR in manual test: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end
