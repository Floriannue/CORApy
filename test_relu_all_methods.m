% Test ReLU polynomial approximation for all methods and orders
clear; clc;

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
orders = [1, 2, 3];
methods = ["regression", "ridgeregression"];

fprintf('Testing ReLU polynomial approximation for all methods and orders:\n');
fprintf('l = %g, u = %g\n', l, u);

for method = methods
    fprintf('\n=== Method: %s ===\n', method);
    
    for order = orders
        fprintf('\nOrder %d:\n', order);
        
        % Call computeApproxPoly
        [coeffs, d] = layer.computeApproxPoly(l, u, order, method);
        
        fprintf('coeffs = [');
        for i = 1:length(coeffs)
            if i == length(coeffs)
                fprintf('%g', coeffs(i));
            else
                fprintf('%g, ', coeffs(i));
            end
        end
        fprintf(']\n');
        fprintf('d = %g\n', d);
        
        % Test evaluation at some points
        x_test = linspace(l, u, 10);
        y_true = layer.f(x_test);
        y_approx = polyval(coeffs, x_test);
        
        fprintf('Sample evaluations:\n');
        for i = 1:min(5, length(x_test))
            fprintf('  x = %g: true = %g, approx = %g\n', x_test(i), y_true(i), y_approx(i));
        end
    end
end
