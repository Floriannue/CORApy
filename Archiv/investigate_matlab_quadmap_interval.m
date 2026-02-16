% investigate_matlab_quadmap_interval - Investigate how MATLAB's quadMap handles Interval

% Create test case matching actual usage
Z = zonotope([1, 1, 0.5; 0, 0, 0.3]);
H_interval = interval([-0.1, -0.05; -0.05, -0.1], [0.1, 0.05; 0.05, 0.1]);
H = {H_interval};

fprintf('Investigating MATLAB quadMap with Interval:\n');
fprintf('Z: center = %s, generators shape = %s\n', mat2str(center(Z)), mat2str(size(generators(Z))));
fprintf('H{1} type: %s\n', class(H{1}));

% Test what happens in quadMap
Zmat = [center(Z), generators(Z)];
fprintf('\nZmat shape: %s\n', mat2str(size(Zmat)));

% Test: Qnonempty(i) = any(Q{i}(:))
fprintf('\nTesting any(Q{i}(:)):\n');
try
    Q_flat = H{1}(:);
    fprintf('  H{1}(:) type: %s\n', class(Q_flat));
    if isa(Q_flat, 'interval')
        fprintf('  H{1}(:) is Interval vector\n');
        fprintf('  Length: %d\n', length(Q_flat));
        fprintf('  First element: [%.6f, %.6f]\n', Q_flat(1).inf, Q_flat(1).sup);
    end
    
    Qnonempty = any(H{1}(:));
    fprintf('  any(H{1}(:)) result: %d (type: %s)\n', Qnonempty, class(Qnonempty));
catch ME
    fprintf('  ERROR: %s\n', ME.message);
end

% Test: quadMat = Zmat'*Q{i}*Zmat
fprintf('\nTesting matrix multiplication:\n');
try
    quadMat = Zmat' * H{1} * Zmat;
    fprintf('  quadMat type: %s\n', class(quadMat));
    fprintf('  quadMat size: %s\n', mat2str(size(quadMat)));
    
    if isa(quadMat, 'interval')
        fprintf('  quadMat is Interval matrix\n');
        fprintf('  quadMat(1,1) = [%.6f, %.6f]\n', quadMat(1,1).inf, quadMat(1,1).sup);
        
        % Test what MATLAB does when extracting diagonal
        fprintf('\n  Testing diagonal extraction:\n');
        quadMat_sub = quadMat(2:3, 2:3);
        fprintf('    quadMat(2:3,2:3) type: %s\n', class(quadMat_sub));
        
        if isa(quadMat_sub, 'interval')
            diag_vals = diag(quadMat_sub);
            fprintf('    diag(quadMat_sub) type: %s\n', class(diag_vals));
            fprintf('    diag_vals(1) = [%.6f, %.6f]\n', diag_vals(1).inf, diag_vals(1).sup);
            
            % Try to see what happens when we try to assign
            fprintf('\n    Testing assignment:\n');
            G_test = zeros(2, 1);
            try
                % This should fail based on previous test
                G_test(1:2) = 0.5 * diag(quadMat_sub);
                fprintf('      Assignment succeeded! G_test = %s\n', mat2str(G_test));
            catch ME2
                fprintf('      Assignment failed: %s\n', ME2.message);
                fprintf('      This confirms MATLAB cannot assign Interval to numeric\n');
            end
        end
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s at line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end

% Check if there's a conversion method
fprintf('\nChecking for conversion methods:\n');
try
    H_numeric = double(H{1});
    fprintf('  double(H{1}) works! Type: %s\n', class(H_numeric));
    fprintf('  double(H{1}) max: %.6f\n', max(abs(H_numeric(:))));
catch ME3
    fprintf('  double(H{1}) failed: %s\n', ME3.message);
end

try
    H_sup = supremum(H{1});
    fprintf('  supremum(H{1}) works! Type: %s\n', class(H_sup));
    fprintf('  supremum(H{1}) max: %.6f\n', max(abs(H_sup(:))));
catch ME4
    fprintf('  supremum(H{1}) failed: %s\n', ME4.message);
end
