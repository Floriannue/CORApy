% Debug script to test contractPoly with forwardBackward
% Testing: x[0]^2 + x[1]^2 - 4 = 0
% Initial domain: [1, 3] for both variables

% Setup
dim = 2;
dom = interval([1; 1], [3; 3]);

% Define polynomial: x[0]^2 + x[1]^2 - 4 = 0
% This is: x[0]^2 + x[1]^2 = 4
% For domain [1,3] x [1,3]: min = 1^2 + 1^2 = 2, max = 3^2 + 3^2 = 18
% So constraint [0, 0] should contract to where x[0]^2 + x[1]^2 = 4
% This means: x[0]^2 + x[1]^2 ∈ [4, 4]
% For x[0] ∈ [1, 3], x[1] ∈ [1, 3]: 
%   If x[0] = 1, then x[1]^2 = 3, so x[1] = sqrt(3) ≈ 1.732
%   If x[0] = 2, then x[1]^2 = 0, so x[1] = 0 (but x[1] ∈ [1,3], so impossible)
%   If x[0] = sqrt(2), then x[1]^2 = 2, so x[1] = sqrt(2) ≈ 1.414 (but x[1] ∈ [1,3], so possible)
% Actually, the constraint x[0]^2 + x[1]^2 = 4 with x[0], x[1] ∈ [1,3] means:
%   x[0]^2 ∈ [1, 9], x[1]^2 ∈ [1, 9]
%   x[0]^2 + x[1]^2 ∈ [2, 18]
%   For x[0]^2 + x[1]^2 = 4: x[0]^2 ∈ [1, 3], x[1]^2 ∈ [1, 3]
%   So x[0] ∈ [1, sqrt(3)], x[1] ∈ [1, sqrt(3)]

% Define polynomial using polynomial representation
% f(x) = x[0]^2 + x[1]^2 - 4
% In polynomial form: c + sum(G(:,i) * prod(x.^E(:,i))) + GI * x_indep
% For x[0]^2: E = [2, 0], G = [1]
% For x[1]^2: E = [0, 2], G = [1]
% Constant: c = -4

% Create polynomial function
c = -4;
G = [1, 1];  % Coefficients for x[0]^2 and x[1]^2
E = [2, 0;    % Exponent for x[0]^2
     0, 2];   % Exponent for x[1]^2
GI = [];      % No independent generators
x_indep = []; % No independent variables

% Create function handle
f = @(x) c + G(1) * x(1)^E(1,1) * x(2)^E(2,1) + G(2) * x(1)^E(1,2) * x(2)^E(2,2);

% Test contractPoly with forwardBackward
fprintf('=== Testing contractPoly with forwardBackward ===\n');
fprintf('Initial domain:\n');
display(dom);

% Call contractPoly
dom_contracted = contractPoly(f, dom, 'forwardBackward');

fprintf('\nContracted domain:\n');
if isempty(dom_contracted)
    fprintf('Empty set (None)\n');
else
    display(dom_contracted);
end

% Also test with syntax tree to see the structure
fprintf('\n=== Testing with syntax tree ===\n');
% Build syntax tree manually to understand structure
x1 = interval([1; 1], [3; 3]);
x2 = [];

% Build: (-4) + x[0]^2 + x[1]^2
% This should group as: ((-4) + x[0]^2) + x[1]^2
fprintf('Building syntax tree for: (-4) + x[0]^2 + x[1]^2\n');

% Test backpropagation with constraint [0, 0]
constraint = interval(0, 0);
fprintf('Backpropagating constraint [0, 0] through the tree\n');
