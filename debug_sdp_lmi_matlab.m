% debug_sdp_lmi_matlab.m
addpath(genpath('D:\Bachelorarbeit\Translate_Cora\cora_matlab'));

% Define ellipsoids as in test_ellipsoid_or.py
E1 = ellipsoid([5.43878115, 12.49771836; 12.49771836, 29.66621173], [-0.74450683; 3.58006475]);
E0_original = ellipsoid([0.0, 0.0; 0.0, 0.0], [1.09869336; -1.98843878]);

% Define placeholder values for A2, bt, l for comparison
n = dim(E1); % Dimension
N = 2; % Number of ellipsoids

A2_placeholder = eye(n); % Identity matrix for A2
bt_placeholder = zeros(n,1); % Zero vector for bt
l_placeholder = ones(N,1); % Vector of ones for l

fprintf('--- MATLAB LMI Blocks Debug ---\n');
fprintf('Using A2_placeholder=\n'); disp(A2_placeholder);
fprintf('bt_placeholder=\n'); disp(bt_placeholder);
fprintf('l_placeholder=\n'); disp(l_placeholder);

% Manual bloating simulation (as in priv_orEllipsoidOA.m)
max_val_e1 = max(svd(E1.Q));
max_val_e0 = max(svd(E0_original.Q));
max_val = max(max_val_e1, max_val_e0);

fac = 0.001; 
th = fac * max_val;
if th == 0, th = fac; end

fprintf('MATLAB Bloating: max_val=%f, th=%f\n', max_val, th);

E_cells_processed = {E1, E0_original};

for i=1:N
    E_i = E_cells_processed{i};
    
    % Apply bloating if degenerate (as in priv_orEllipsoidOA.m)
    if ~isFullDim(E_i)
        nd_i = rank(E_i);
        [Ti,Si_diag_full,~] = svd(E_i.Q);
        si = diag(Si_diag_full);
        Si_bloated = diag([si(1:nd_i); th*ones(n-nd_i,1)]);
        E_i = ellipsoid(Ti*Si_bloated*Ti',E_i.q);
        E_cells_processed{i} = E_i; % Update E_i in the cell array
    end

    Qinv_i = inv(E_i.Q);
    c_i = E_i.q'*Qinv_i*E_i.q - 1;

    % Construct LMI block for current ellipsoid using placeholder values
    M11 = -A2_placeholder + l_placeholder(i) * Qinv_i;
    M12 = -bt_placeholder - l_placeholder(i) * Qinv_i * E_i.q;
    M13 = zeros(n, n);

    M21 = (-bt_placeholder - l_placeholder(i) * Qinv_i * E_i.q)';
    M22 = 1 + l_placeholder(i) * c_i;
    M23 = -bt_placeholder';

    M31 = zeros(n, n);
    M32 = -bt_placeholder;
    M33 = A2_placeholder;

    fprintf('\nEllipsoid %d LMI Blocks:\n', i-1); % Python is 0-indexed
    fprintf('  M11 (-A2 + l[i]*Qinv_i):\n'); disp(M11);
    fprintf('  M12 (-bt - l[i]*Qinv_i@q_i):\n'); disp(M12);
    fprintf('  M13 (zeros(n)):\n'); disp(M13);
    fprintf('  M21 ((-bt - l[i]*Qinv_i@q_i).T):\n'); disp(M21);
    fprintf('  M22 (1 + l[i]*c_i):\n'); disp(M22);
    fprintf('  M23 (-bt.T):\n'); disp(M23);
    fprintf('  M31 (zeros(n)):\n'); disp(M31);
    fprintf('  M32 (-bt):\n'); disp(M32);
    fprintf('  M33 (A2):\n'); disp(M33);
end 