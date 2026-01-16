% Debug script to check dimensions in MATLAB
A = [-1, 0; 0, -2];
B = [1; 1];
uTrans = [0.1];  % Row vector
state_dim = size(A, 1);

% Create integrator system
% A_int = [A, uTrans; zeros(1, state_dim), 0];
% Note: uTrans needs to be column vector for concatenation
% For this debug script, we'll skip A_int construction

% Equivalent initial state
eqivState = [zeros(state_dim, 1); 1];

% Options
options.krylovError = 1e-6;
options.krylovOrder = 15;
options.krylovStep = 5;

% Arnoldi (simplified - would need actual function)
% For debugging, let's assume V_uT has some dimensions
% V_uT should be (state_dim+1, k) where k is Krylov dimension
k = 5;  % Example Krylov dimension
V_uT = randn(state_dim+1, k);
H_uT = randn(k, k);

fprintf('V_uT shape: (%d, %d)\n', size(V_uT, 1), size(V_uT, 2));
fprintf('V_uT(1:state_dim,:) shape: (%d, %d)\n', state_dim, size(V_uT, 2));
fprintf('H_uT shape: (%d, %d)\n', size(H_uT, 1), size(H_uT, 2));

% Create reduced system
C = [1, 0];
V_uT_proj = C * V_uT(1:state_dim, :);
fprintf('V_uT_proj shape: (%d, %d)\n', size(V_uT_proj, 1), size(V_uT_proj, 2));

% G should have shape (k, k) where k is size of H_uT
% For now, create a dummy G (ensure lower <= upper)
G_lb = randn(k, k);
G_ub = G_lb + abs(randn(k, k));  % Ensure ub >= lb
G = interval(G_lb, G_ub);
fprintf('G shape: (%d, %d)\n', size(G.inf, 1), size(G.inf, 2));
fprintf('G(:,1) shape: (%d, %d)\n', size(G.inf, 1), 1);

% The multiplication
inputCorr_unprojected = V_uT(1:state_dim, :) * G(:, 1);
fprintf('inputCorr_unprojected shape: (%d, %d)\n', size(inputCorr_unprojected.inf, 1), size(inputCorr_unprojected.inf, 2));
