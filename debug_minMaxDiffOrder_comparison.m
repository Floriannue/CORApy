% Debug script to compare minMaxDiffOrder between MATLAB and Python
% This will help us understand the exact difference in computation

% Create ReLU layer
layer = nnReLULayer();

% Test parameters
l = -1;
u = 1;
order = 2;

% Get the coefficients from regression
numPoints = 10*(order+1);
x = linspace(l, u, numPoints);
y = layer.f(x);
coeffs = nnHelper.leastSquarePolyFunc(x, y, order);

% Get derivative bounds
[df_l, df_u] = layer.getDerBounds(l, u);

fprintf('=== MATLAB minMaxDiffOrder Debug ===\n');
fprintf('l: %.1f, u: %.1f\n', l, u);
fprintf('coeffs: [%.8f, %.8f, %.8f]\n', coeffs(1), coeffs(2), coeffs(3));
fprintf('df_l: %.8f, df_u: %.8f\n', df_l, df_u);

% Call minMaxDiffOrder
[diffl, diffu] = nnHelper.minMaxDiffOrder(coeffs, l, u, layer.f, df_l, df_u);

fprintf('minMaxDiffOrder results:\n');
fprintf('  diffl: %.8f\n', diffl);
fprintf('  diffu: %.8f\n', diffu);

% Compute final values
diffc = (diffl + diffu) / 2;
coeffs_final = coeffs;
coeffs_final(end) = coeffs_final(end) + diffc;
d_final = diffu - diffc;

fprintf('Final computation:\n');
fprintf('  diffc: %.8f\n', diffc);
fprintf('  coeffs_final: [%.8f, %.8f, %.8f]\n', coeffs_final(1), coeffs_final(2), coeffs_final(3));
fprintf('  d_final: %.8f\n', d_final);

% Let's also test the getDerInterval function
[der2l, der2u] = nnHelper.getDerInterval(coeffs, l, u);
fprintf('\ngetDerInterval results:\n');
fprintf('  der2l: %.8f\n', der2l);
fprintf('  der2u: %.8f\n', der2u);

% Test the derivative calculation
der = max(abs([
    df_l - -der2l;
    df_l - -der2u;
    df_u - -der2l;
    df_u - -der2u;
]));
fprintf('  der: %.8f\n', der);
