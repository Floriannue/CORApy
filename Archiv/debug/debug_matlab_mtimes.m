% Debug script to capture MATLAB input/output pairs for mtimes
% Covers interval, zonotope, and matZonotope cases used in Python tests

outputFile = fullfile(pwd, 'matlab_mtimes_debug.txt');
fid = fopen(outputFile, 'w');
if fid == -1
    error('Failed to open output file: %s', outputFile);
end
fprintf(fid, 'MATLAB mtimes debug output\n');

print_interval = @(name, I) fprintf(fid, '%s class=interval inf=%s sup=%s size_inf=%s size_sup=%s\n', name, ...
    mat2str(infimum(I), 15), mat2str(supremum(I), 15), mat2str(size(infimum(I))), mat2str(size(supremum(I))));

print_zonotope = @(name, Z) fprintf(fid, '%s class=zonotope c=%s G=%s size_c=%s size_G=%s empty=%d\n', name, ...
    mat2str(Z.c, 15), mat2str(Z.G, 15), mat2str(size(Z.c)), mat2str(size(Z.G)), representsa(Z,'emptySet'));

print_matZonotope = @(name, MZ) fprintf(fid, '%s class=matZonotope C=%s size_C=%s size_G=%s\n', name, ...
    mat2str(MZ.C, 15), mat2str(size(MZ.C)), mat2str(size(MZ.G)));

%% Interval mtimes cases (match Python tests)
a = interval(-1,2); b = 3; c = a * b; print_interval('int_scalar_1', c);
a = -2; b = interval(2,4); c = a * b; print_interval('int_scalar_2', c);
a = interval(-2,1); b = interval(2,4); c = a * b; print_interval('int_scalar_3', c);

a = interval(-1,2); b = [-1 0; 1 2]; c = a * b; print_interval('int_scalar_matrix_1', c);
a = -2; b = interval([2 -3; -1 2],[4 -2; 1 3]); c = a * b; print_interval('int_scalar_matrix_2', c);
a = interval(-1,2); b = interval([2 -3; -1 2],[4 -2; 1 3]); c = a * b; print_interval('int_scalar_matrix_3', c);

a = interval([-1 0; -2 2],[2 1; -1 3]); b = -1; c = a * b; print_interval('int_matrix_scalar_1', c);
a = [-1 0; 1 2]; b = interval(-2,1); c = a * b; print_interval('int_matrix_scalar_2', c);
a = interval([-1 0; -2 2],[2 1; -1 3]); b = interval(-2,1); c = a * b; print_interval('int_matrix_scalar_3', c);

a = interval([-1 0; -2 2],[2 1; -1 3]); b = eye(2); c = a * b; print_interval('int_matrix_matrix_1', c);
a = [2 0; 0 2]; b = interval([-1 0; -2 2],[2 1; -1 3]); c = a * b; print_interval('int_matrix_matrix_2', c);
a = interval([1 0; 0 1],[2 1; 1 2]); b = interval([-1 0; 0 -1],[1 1; 1 1]); c = a * b; print_interval('int_matrix_matrix_3', c);

a = interval(0,0); b = interval(-inf,inf); c = a * b; print_interval('int_zero_scalar', c);
a = interval(zeros(2), zeros(2)); b = interval([-1 1; 2 -2],[1 2; 3 -1]); c = a * b; print_interval('int_zero_matrix', c);

%% Zonotope mtimes cases (match Python tests)
Z = zonotope.empty(2);
M = [-1 2; 3 -4]; Zm = M * Z; print_zonotope('zono_empty_square', Zm);
M = [-1 2]; Zm = M * Z; print_zonotope('zono_empty_proj', Zm);
M = [-1 2; 3 -4; 5 6]; Zm = M * Z; print_zonotope('zono_empty_lift', Zm);

Z = zonotope([-4; 1], [-3 -2 -1; 2 3 4]);
M = [-1 2; 3 -4]; Zm = M * Z; print_zonotope('zono_full_square', Zm);
M = [-1 2]; Zm = M * Z; print_zonotope('zono_full_proj', Zm);
M = [-1 2; 1 0; 0 1]; Zm = M * Z; print_zonotope('zono_full_lift', Zm);
M = 2; Zm = Z * M; print_zonotope('zono_scalar_right', Zm);

%% matZonotope mtimes (basic coverage)
matZ = matZonotope([1 0; 0 1], cat(3, [0.1 0; 0 0.1]));
Z = zonotope([1; 2], [0.5 0; 0 0.5]);
Z_m = matZ * Z; print_zonotope('matZ_times_zono', Z_m);

fclose(fid);
fprintf('Wrote %s\n', outputFile);
