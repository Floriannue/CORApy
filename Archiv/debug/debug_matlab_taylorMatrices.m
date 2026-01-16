% Debug script to verify taylorMatrices test against MATLAB
% This generates exact input/output pairs for Python tests

% tolerance
tol = 1e-14;

fprintf('=== Test 1: Basic Functionality ===\n\n');

% Setup: Create a simple linear system (matching Python test)
A = [-1, 0; 0, -2];
B = [1; 1];
sys = linearSys(A, B);

fprintf('Input:\n');
fprintf('A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('B = [%g; %g]\n', B(1), B(2));

timeStep = 0.1;
truncationOrder = 10;

fprintf('timeStep = %g\n', timeStep);
fprintf('truncationOrder = %d\n', truncationOrder);
fprintf('tol = %g\n', tol);

% Execute
[E, F, G] = taylorMatrices(sys, timeStep, truncationOrder);

fprintf('\n=== Output Results ===\n');

% 1. E should be a matrix
fprintf('\nE (remainder matrix):\n');
fprintf('E type: %s\n', class(E));
fprintf('E shape: %dx%d\n', size(E,1), size(E,2));
if isa(E, 'interval')
    fprintf('E.inf = \n');
    disp(E.inf);
    fprintf('E.sup = \n');
    disp(E.sup);
    fprintf('E.center() = \n');
    disp(center(E));
    fprintf('E.rad() = \n');
    disp(rad(E));
else
    fprintf('E = \n');
    disp(E);
end

% 2. F should be a matrix
fprintf('\nF (correction matrix for state):\n');
fprintf('F type: %s\n', class(F));
fprintf('F shape: %dx%d\n', size(F,1), size(F,2));
if isa(F, 'interval')
    fprintf('F.inf = \n');
    disp(F.inf);
    fprintf('F.sup = \n');
    disp(F.sup);
    fprintf('F.center() = \n');
    disp(center(F));
else
    fprintf('F = \n');
    disp(F);
end

% 3. G should be a matrix
fprintf('\nG (correction matrix for input):\n');
fprintf('G type: %s\n', class(G));
fprintf('G shape: %dx%d\n', size(G,1), size(G,2));
if isa(G, 'interval')
    fprintf('G.inf = \n');
    disp(G.inf);
    fprintf('G.sup = \n');
    disp(G.sup);
    fprintf('G.center() = \n');
    disp(center(G));
else
    fprintf('G = \n');
    disp(G);
end

% 4. System should have taylor object
fprintf('\nSystem taylor object:\n');
fprintf('hasattr taylor: %d\n', ~isempty(sys.taylor));
fprintf('taylor type: %s\n', class(sys.taylor));

% Save to file for Python test
fid = fopen('taylorMatrices_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB taylorMatrices Test Output\n');
fprintf(fid, '==================================\n\n');
fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'B = [%g; %g]\n', B(1), B(2));
fprintf(fid, 'timeStep = %g\n', timeStep);
fprintf(fid, 'truncationOrder = %d\n', truncationOrder);
fprintf(fid, 'tol = %g\n', tol);
fprintf(fid, '\nOutput:\n');
fprintf(fid, 'E type: %s\n', class(E));
fprintf(fid, 'E shape: %dx%d\n', size(E,1), size(E,2));
if isa(E, 'interval')
    E_inf = E.inf;
    E_sup = E.sup;
    E_center = center(E);
    fprintf(fid, 'E.inf = [%.15g, %.15g; %.15g, %.15g]\n', E_inf(1,1), E_inf(1,2), E_inf(2,1), E_inf(2,2));
    fprintf(fid, 'E.sup = [%.15g, %.15g; %.15g, %.15g]\n', E_sup(1,1), E_sup(1,2), E_sup(2,1), E_sup(2,2));
    fprintf(fid, 'E.center() = [%.15g, %.15g; %.15g, %.15g]\n', E_center(1,1), E_center(1,2), E_center(2,1), E_center(2,2));
else
    fprintf(fid, 'E = [%.15g, %.15g; %.15g, %.15g]\n', E(1,1), E(1,2), E(2,1), E(2,2));
end
fprintf(fid, 'F type: %s\n', class(F));
fprintf(fid, 'F shape: %dx%d\n', size(F,1), size(F,2));
if isa(F, 'interval')
    F_center = center(F);
    fprintf(fid, 'F.center() = [%.15g; %.15g]\n', F_center(1), F_center(2));
else
    fprintf(fid, 'F = [%.15g; %.15g]\n', F(1), F(2));
end
fprintf(fid, 'G type: %s\n', class(G));
fprintf(fid, 'G shape: %dx%d\n', size(G,1), size(G,2));
if isa(G, 'interval')
    G_center = center(G);
    fprintf(fid, 'G.center() = [%.15g; %.15g]\n', G_center(1), G_center(2));
else
    fprintf(fid, 'G = [%.15g; %.15g]\n', G(1), G(2));
end
fprintf(fid, 'hasattr taylor: %d\n', ~isempty(sys.taylor));
fprintf(fid, 'taylor type: %s\n', class(sys.taylor));
fclose(fid);

fprintf('\nResults saved to taylorMatrices_matlab_output.txt\n');
