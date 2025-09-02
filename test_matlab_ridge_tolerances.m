% Test MATLAB ridge regression tolerances for different lambda values
clear; clc;

fprintf('Testing MATLAB ridge regression tolerances:\n\n');

% Test with noisy data similar to Python test
rng(1); % Set seed for reproducibility

for order = 1:3
    fprintf('=== Order %d ===\n', order);
    
    % Generate exact data points (same as Python)
    x = rand(order + 1, 1);
    y = rand(order + 1, 1);
    
    % Add noise (same as Python)
    n = 50;
    noise = rand(n * (order + 1), 1) * 0.02 - 0.01; % uniform(-0.01, 0.01)
    y_noisy = repmat(y, n, 1) + noise;
    x_repeated = repmat(x, n, 1);
    
    % Convert to row vectors for MATLAB function
    x_repeated = x_repeated';
    y_noisy = y_noisy';
    
    fprintf('Original y: [%.6f', y(1));
    for i = 2:length(y)
        fprintf(', %.6f', y(i));
    end
    fprintf(']\n');
    
    % Test with different lambda values
    lambda_values = [0.001, 0.01, 0.1];
    
    for lambda_reg = lambda_values
        fprintf('\nLambda = %.3f:\n', lambda_reg);
        
        % Get coefficients
        coeffs = nnHelper.leastSquareRidgePolyFunc(x_repeated, y_noisy, order, lambda_reg);
        
        % Evaluate at original x points
        y_poly = polyval(coeffs, x);
        
        % Calculate differences
        diff = abs(y - y_poly);
        max_diff = max(diff);
        mean_diff = mean(diff);
        
        fprintf('  Max difference: %.6f\n', max_diff);
        fprintf('  Mean difference: %.6f\n', mean_diff);
        fprintf('  Differences: [%.6f', diff(1));
        for i = 2:length(diff)
            fprintf(', %.6f', diff(i));
        end
        fprintf(']\n');
        
        % Suggest tolerance
        suggested_tol = max_diff * 1.1; % 10% margin
        fprintf('  Suggested tolerance: %.6f\n', suggested_tol);
    end
    fprintf('\n');
end

fprintf('Done.\n');
