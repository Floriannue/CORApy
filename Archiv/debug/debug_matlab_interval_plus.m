% Debug script to capture MATLAB input/output pairs for interval plus
% Generates expected interval bounds for Python tests

outputFile = 'matlab_interval_plus_output.txt';
fid = fopen(outputFile, 'w');
fprintf(fid, 'MATLAB interval plus debug output\n');

print_interval = @(name, I) fprintf(fid, '%s class=interval inf=%s sup=%s\n', name, ...
    mat2str(infimum(I), 15), mat2str(supremum(I), 15));

print_zonotope = @(name, Z) fprintf(fid, '%s class=zonotope c=%s G=%s size_c=%s size_G=%s empty=%d\n', name, ...
    mat2str(Z.c, 15), mat2str(Z.G, 15), mat2str(size(Z.c)), mat2str(size(Z.G)), representsa(Z,'emptySet'));

print_empty = @(name, I) fprintf(fid, '%s empty=%d\n', name, representsa(I, 'emptySet'));

print_set = @(name, S) ...
    (isa(S,'interval') && print_interval(name, S)) || ...
    (isa(S,'zonotope') && print_zonotope(name, S)) || ...
    fprintf(fid, '%s class=%s\n', name, class(S));

% empty
I = interval.empty(1);
v = 1;
I_plus = I + v;
print_empty('case_empty_I_plus', I_plus);
I_plus = v + I;
print_empty('case_empty_v_plus', I_plus);

% bounded interval, numeric
v = [2;1];
I = interval([-2;-1],[3; 4]);
I_plus = v + I;
print_interval('case_bounded_numeric_v_plus', I_plus);
I_plus = I + v;
print_interval('case_bounded_numeric_I_plus', I_plus);

% unbounded interval, numeric
I = interval(-Inf,2);
v = 1;
I_plus = I + v;
print_interval('case_unbounded_numeric_I_plus', I_plus);
I_plus = v + I;
print_interval('case_unbounded_numeric_v_plus', I_plus);

% bounded interval, bounded interval
I1 = interval([-2;-1],[3;4]);
I2 = interval([-1;-3],[1;-1]);
I_plus = I1 + I2;
print_interval('case_bounded_interval_I1_plus_I2', I_plus);
I_plus = I2 + I1;
print_interval('case_bounded_interval_I2_plus_I1', I_plus);

% unbounded interval, unbounded interval
I1 = interval([-Inf;-2],[2;4]);
I2 = interval([-1;0],[1;Inf]);
I_plus = I1 + I2;
print_interval('case_unbounded_interval_I1_plus_I2', I_plus);
I_plus = I2 + I1;
print_interval('case_unbounded_interval_I2_plus_I1', I_plus);

% interval matrix, numeric
I = interval([-2 -1; 0 2],[3 5; 2 3]);
v = 2;
I_plus = I + v;
print_interval('case_interval_matrix_I_plus', I_plus);
I_plus = v + I;
print_interval('case_interval_matrix_v_plus', I_plus);

% scalar operations
I = interval([-1;0],[2;3]);
v = 5;
I_plus = I + v;
print_interval('case_scalar_I_plus', I_plus);
I_plus = v + I;
print_interval('case_scalar_v_plus', I_plus);

% vector operations
I = interval([-1;0],[2;3]);
v = [1;-1];
I_plus = I + v;
print_interval('case_vector_I_plus', I_plus);
I_plus = v + I;
print_interval('case_vector_v_plus', I_plus);

% matrix operations (interval + interval)
I1 = interval([-1 0; 1 -1],[2 1; 3 2]);
I2 = interval([0 -1; -1 0],[1 0; 2 1]);
I_plus = I1 + I2;
print_interval('case_matrix_I1_plus_I2', I_plus);

% zero operations
I = interval([-1;0],[2;3]);
v = 0;
I_plus = I + v;
print_interval('case_zero_scalar_I_plus', I_plus);
v = zeros(2,1);
I_plus = I + v;
print_interval('case_zero_vector_I_plus', I_plus);

% associativity
I1 = interval([-1;0],[1;2]);
I2 = interval([0;-1],[2;1]);
I3 = interval([-1;-1],[1;1]);
I_plus = (I1 + I2) + I3;
print_interval('case_assoc_left', I_plus);
I_plus = I1 + (I2 + I3);
print_interval('case_assoc_right', I_plus);

% interval + zonotope (result should be zonotope per MATLAB precedence)
I = interval([-1;-1],[1;1]);
Z = zonotope([0;0],[1 0;0 1]);
S = I + Z;
print_set('case_interval_plus_zonotope', S);
S = Z + I;
print_set('case_zonotope_plus_interval', S);

% interval + zonotope different shapes
I = interval([0;-2],[2;1]);
Z = zonotope([1;-1],[0.5 0;0 0.5]);
S = I + Z;
print_set('case_interval_plus_zonotope_diff', S);

% interval + zonotope 1d
I = interval(-2,3);
Z = zonotope(1,2);
S = I + Z;
print_set('case_interval_plus_zonotope_1d', S);

% interval + zonotope no generators (point)
I = interval([-1;0],[1;2]);
Z = zonotope([2;1],[]);
S = I + Z;
print_set('case_interval_plus_zonotope_point', S);

% interval + zonotope empty
I = interval.empty(2);
Z = zonotope([0;0],[1 0;0 1]);
S = I + Z;
print_set('case_interval_plus_zonotope_empty', S);
S = Z + I;
print_set('case_zonotope_plus_interval_empty', S);

fclose(fid);
fprintf('Wrote %s\n', outputFile);
