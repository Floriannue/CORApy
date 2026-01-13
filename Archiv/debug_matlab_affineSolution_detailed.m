% Detailed debug script to trace affineSolution computation step-by-step
% tolerance
tol = 1e-14;

% init system, state, input, and algorithm parameters
A = [-1 -4; 4 -1];
sys = linearSys(A);
X = zonotope([40;20],[1 4 2; -1 3 5]);
u = [2;-1];
timeStep = 0.05;
truncationOrder = 6;

fprintf('=== Step-by-step computation ===\n\n');

% Check if Ainv exists
Ainv = getTaylor(sys,'Ainv');
fprintf('Ainv exists: %d\n', ~isempty(Ainv));
if ~isempty(Ainv)
    fprintf('Ainv = \n');
    disp(Ainv);
end

% Get eAdt
eAdt = getTaylor(sys,'eAdt',struct('timeStep',timeStep));
fprintf('\neAdt = \n');
disp(eAdt);

% Check analytical computation
Ainv_analytical = inv(A);
eAdt_analytical = expm(A*timeStep);
fprintf('\nAinv (analytical) = \n');
disp(Ainv_analytical);
fprintf('eAdt (analytical) = \n');
disp(eAdt_analytical);

% Compare Ainv
if ~isempty(Ainv)
    diff_Ainv = abs(Ainv - Ainv_analytical);
    fprintf('\nMax difference in Ainv: %.15g\n', max(max(diff_Ainv)));
end

% Compare eAdt
diff_eAdt = abs(eAdt - eAdt_analytical);
fprintf('Max difference in eAdt: %.15g\n', max(max(diff_eAdt)));

% Compute (eAdt - eye)
eye_mat = eye(2);
eAdt_minus_eye = eAdt - eye_mat;
fprintf('\neAdt - eye = \n');
disp(eAdt_minus_eye);

% Compute Ainv * (eAdt - eye)
if ~isempty(Ainv)
    Ainv_eAdt_minus_eye = Ainv * (eAdt - eye_mat);
    fprintf('\nAinv * (eAdt - eye) = \n');
    disp(Ainv_eAdt_minus_eye);
    
    % Compare with analytical
    Ainv_eAdt_minus_eye_analytical = Ainv_analytical * (eAdt_analytical - eye_mat);
    fprintf('Ainv * (eAdt - eye) (analytical) = \n');
    disp(Ainv_eAdt_minus_eye_analytical);
    diff_matrix = abs(Ainv_eAdt_minus_eye - Ainv_eAdt_minus_eye_analytical);
    fprintf('Max difference in matrix: %.15g\n', max(max(diff_matrix)));
    
    % Compute Pu
    Pu_computed = Ainv_eAdt_minus_eye * u;
    fprintf('\nPu (computed step-by-step) = \n');
    disp(Pu_computed);
    
    % Analytical Pu
    Pu_analytical = Ainv_analytical * (eAdt_analytical - eye_mat) * u;
    fprintf('Pu (analytical) = \n');
    disp(Pu_analytical);
    diff_Pu = abs(Pu_computed - Pu_analytical);
    fprintf('Max difference in Pu: %.15g\n', max(diff_Pu));
end

% Now call affineSolution
fprintf('\n=== Calling affineSolution ===\n');
[Htp,Pu,Hti,C_state,C_input] = ...
    affineSolution(sys,X,u,timeStep,truncationOrder);
fprintf('Pu from affineSolution = \n');
disp(Pu);
fprintf('Pu type: %s\n', class(Pu));

% Compare
Pu_true = inv(A) * (expm(A*timeStep) - eye(2)) * u;
fprintf('\nPu_true = \n');
disp(Pu_true);
diff = abs(Pu - Pu_true);
fprintf('Max difference: %.15g\n', max(diff));
fprintf('Difference = \n');
disp(diff);
