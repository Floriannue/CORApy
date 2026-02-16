% Test MATLAB's max() behavior with Inf and NaN
% This verifies how MATLAB handles these special values

fprintf('=== Testing MATLAB max() behavior ===\n\n');

% Test 1: max with Inf
fprintf('Test 1: max([Inf, 5, 3])\n');
result1 = max([Inf, 5, 3]);
fprintf('  Result: %.6e\n', result1);
fprintf('  isinf(result): %d\n', isinf(result1));
fprintf('  isnan(result): %d\n', isnan(result1));
fprintf('\n');

% Test 2: max with NaN
fprintf('Test 2: max([NaN, 5, 3])\n');
result2 = max([NaN, 5, 3]);
fprintf('  Result: %.6e\n', result2);
fprintf('  isinf(result): %d\n', isinf(result2));
fprintf('  isnan(result): %d\n', isnan(result2));
fprintf('\n');

% Test 3: max with both Inf and NaN
fprintf('Test 3: max([Inf, NaN, 5])\n');
result3 = max([Inf, NaN, 5]);
fprintf('  Result: %.6e\n', result3);
fprintf('  isinf(result): %d\n', isinf(result3));
fprintf('  isnan(result): %d\n', isnan(result3));
fprintf('\n');

% Test 4: Division producing Inf
fprintf('Test 4: max([1, 2, 3] ./ [0, 1, 1])\n');
result4 = max([1, 2, 3] ./ [0, 1, 1]);
fprintf('  Result: %.6e\n', result4);
fprintf('  isinf(result): %d\n', isinf(result4));
fprintf('  isnan(result): %d\n', isnan(result4));
fprintf('\n');

% Test 5: Division producing NaN
fprintf('Test 5: max([0, 2, 3] ./ [0, 1, 1])\n');
result5 = max([0, 2, 3] ./ [0, 1, 1]);
fprintf('  Result: %.6e\n', result5);
fprintf('  isinf(result): %d\n', isinf(result5));
fprintf('  isnan(result): %d\n', isnan(result5));
fprintf('\n');

% Test 6: Comparison with Inf
fprintf('Test 6: Inf <= 1\n');
result6 = Inf <= 1;
fprintf('  Result: %d\n', result6);
fprintf('\n');

% Test 7: Comparison with NaN
fprintf('Test 7: NaN <= 1\n');
result7 = NaN <= 1;
fprintf('  Result: %d\n', result7);
fprintf('\n');

% Test 8: Inf > Inf
fprintf('Test 8: Inf > Inf\n');
result8 = Inf > Inf;
fprintf('  Result: %d\n', result8);
fprintf('\n');

% Test 9: NaN > NaN
fprintf('Test 9: NaN > NaN\n');
result9 = NaN > NaN;
fprintf('  Result: %d\n', result9);
fprintf('\n');

fprintf('=== Summary ===\n');
fprintf('MATLAB max() with Inf returns: Inf\n');
fprintf('MATLAB max() with NaN returns: NaN\n');
fprintf('Inf <= 1 evaluates to: false (0)\n');
fprintf('NaN <= 1 evaluates to: false (0)\n');
fprintf('Inf > Inf evaluates to: false (0)\n');
fprintf('NaN > NaN evaluates to: false (0)\n');
