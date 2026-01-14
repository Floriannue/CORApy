% Debug script to verify all test values match MATLAB
% Compare against Python test_linearSys_reach.py

clear; close all; clc;

fprintf('=== Test 1: test_reach_basic ===\n');
try
    % Basic reachability test
    A = [-0.1, -2; 2, -0.1];
    B = [1; 0];
    sys = linearSys(A, B);
    
    params.tFinal = 0.2;
    params.R0 = zonotope([1; 1], 0.1*eye(2));
    params.U = zonotope(0, 0.01);
    
    options.timeStep = 0.05;
    options.taylorTerms = 4;
    options.zonotopeOrder = 20;
    options.linAlg = 'standard';
    
    R = sys.reach(params, options);
    
    fprintf('SUCCESS: test_reach_basic\n');
    fprintf('  timePoint sets: %d\n', length(R.timePoint.set));
    fprintf('  timeInterval sets: %d\n', length(R.timeInterval.set));
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

fprintf('\n=== Test 2: test_reach_double_integrator ===\n');
try
    % Double integrator system
    A = zeros(2, 2);
    B = [1; 1];
    sys = linearSys(A, B);
    
    params.tFinal = 1.0;
    params.R0 = zonotope([1; 1], 0.1*eye(2));
    params.U = zonotope(1, 0.1);
    
    options.timeStep = 0.04;
    options.taylorTerms = 4;
    options.zonotopeOrder = 10;
    options.linAlg = 'wrapping-free';
    
    R = sys.reach(params, options);
    
    % Check final interval hull
    final_set = R.timeInterval.set{end};
    IH = interval(final_set);
    
    fprintf('SUCCESS: test_reach_double_integrator\n');
    fprintf('  IH.inf shape: %s\n', mat2str(size(IH.inf)));
    fprintf('  IH.sup shape: %s\n', mat2str(size(IH.sup)));
    fprintf('  IH.inf: [%.2f; %.2f]\n', IH.inf(1), IH.inf(2));
    fprintf('  IH.sup: [%.2f; %.2f]\n', IH.sup(1), IH.sup(2));
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

fprintf('\n=== Test 3: test_reach_wrapping_free ===\n');
try
    A = [-0.1, -2; 2, -0.1];
    B = [1; 0];
    sys = linearSys(A, B);
    
    params.tFinal = 0.2;
    params.R0 = zonotope([1; 1], 0.1*eye(2));
    params.U = zonotope(0, 0.01);
    
    options.timeStep = 0.05;
    options.taylorTerms = 4;
    options.zonotopeOrder = 20;
    options.linAlg = 'wrapping-free';
    
    R = sys.reach(params, options);
    
    fprintf('SUCCESS: test_reach_wrapping_free\n');
    fprintf('  timePoint sets: %d\n', length(R.timePoint.set));
    fprintf('  timeInterval sets: %d\n', length(R.timeInterval.set));
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

fprintf('\n=== Test 4: test_reach_different_algorithms ===\n');
try
    A = [-0.1, -2; 2, -0.1];
    B = [1; 0];
    sys = linearSys(A, B);
    
    params.tFinal = 0.2;
    params.R0 = zonotope([10; 5], 0.5*eye(2));
    params.U = zonotope(0, 0.25);
    
    options.timeStep = 0.05;
    options.taylorTerms = 4;
    options.zonotopeOrder = 20;
    
    % Test standard algorithm
    options.linAlg = 'standard';
    R_standard = sys.reach(params, options);
    
    % Test wrapping-free algorithm
    options.linAlg = 'wrapping-free';
    R_wrappingfree = sys.reach(params, options);
    
    fprintf('SUCCESS: test_reach_different_algorithms\n');
    fprintf('  standard timePoint sets: %d\n', length(R_standard.timePoint.set));
    fprintf('  wrapping-free timePoint sets: %d\n', length(R_wrappingfree.timePoint.set));
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

fprintf('\n=== Test 5: test_reach_with_output_matrix ===\n');
try
    A = [-1, 0; 0, -2];
    B = [1; 1];
    C = [1, 0; 0, 1];
    sys = linearSys(A, B, [], C);
    
    params.tFinal = 0.1;
    params.R0 = zonotope([0; 0], 0.1*eye(2));
    params.U = zonotope(0, 0.01);
    
    options.timeStep = 0.05;
    options.taylorTerms = 4;
    options.zonotopeOrder = 20;
    options.linAlg = 'standard';
    
    R = sys.reach(params, options);
    
    fprintf('SUCCESS: test_reach_with_output_matrix\n');
    fprintf('  timePoint sets: %d\n', length(R.timePoint.set));
    fprintf('  timeInterval sets: %d\n', length(R.timeInterval.set));
catch ME
    fprintf('ERROR: %s\n', ME.message);
end

fprintf('\n=== All tests completed ===\n');
