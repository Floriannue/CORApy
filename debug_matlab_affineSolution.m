% Debug script to verify affineSolution test against MATLAB
% This generates exact input/output pairs for Python tests

% tolerance
tol = 1e-14;

% init system, state, input, and algorithm parameters
A = [-1 -4; 4 -1];
sys = linearSys(A);
X = zonotope([40;20],[1 4 2; -1 3 5]);
u = [2;-1];
timeStep = 0.05;
truncationOrder = 6;

fprintf('=== Input Parameters ===\n');
fprintf('A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf('X center = [%g; %g]\n', X.c(1), X.c(2));
fprintf('X generators shape: %dx%d\n', size(X.G,1), size(X.G,2));
fprintf('u = [%g; %g]\n', u(1), u(2));
fprintf('timeStep = %g\n', timeStep);
fprintf('truncationOrder = %d\n', truncationOrder);
fprintf('tol = %g\n', tol);

% compute reachable sets of first step
[Htp,Pu,Hti,C_state,C_input] = ...
    affineSolution(sys,X,u,timeStep,truncationOrder);

fprintf('\n=== Output Results ===\n');
fprintf('Pu type: %s\n', class(Pu));

% compare particular solution to analytical solution
Pu_true = inv(A)*(expm(A*timeStep) - eye(2)) * u;
fprintf('Pu_true (analytical) = [%.15g; %.15g]\n', Pu_true(1), Pu_true(2));

% Check what Pu is and extract center
if isa(Pu, 'zonotope')
    Pu_center = Pu.c;
    fprintf('Pu.c (computed) = [%.15g; %.15g]\n', Pu_center(1), Pu_center(2));
    diff = Pu_center - Pu_true;
else
    Pu_center = Pu;
    fprintf('Pu (computed) = [%.15g; %.15g]\n', Pu_center(1), Pu_center(2));
    diff = Pu_center - Pu_true;
end

fprintf('Difference = [%.15g; %.15g]\n', diff(1), diff(2));
fprintf('Max absolute difference = %.15g\n', max(abs(diff)));
fprintf('compareMatrices result = %d\n', compareMatrices(Pu,Pu_true,tol,'equal',true));

% Save to file for Python test
fid = fopen('affineSolution_matlab_output.txt', 'w');
fprintf(fid, 'MATLAB affineSolution Test Output\n');
fprintf(fid, '==================================\n\n');
fprintf(fid, 'Input:\n');
fprintf(fid, 'A = [%g, %g; %g, %g]\n', A(1,1), A(1,2), A(2,1), A(2,2));
fprintf(fid, 'X.c = [%g; %g]\n', X.c(1), X.c(2));
fprintf(fid, 'X.G = [%g, %g, %g; %g, %g, %g]\n', X.G(1,1), X.G(1,2), X.G(1,3), X.G(2,1), X.G(2,2), X.G(2,3));
fprintf(fid, 'u = [%g; %g]\n', u(1), u(2));
fprintf(fid, 'timeStep = %g\n', timeStep);
fprintf(fid, 'truncationOrder = %d\n', truncationOrder);
fprintf(fid, 'tol = %g\n', tol);
fprintf(fid, '\nOutput:\n');
fprintf(fid, 'Pu type: %s\n', class(Pu));
if isa(Pu, 'zonotope')
    fprintf(fid, 'Pu.c = [%.15g; %.15g]\n', Pu.c(1), Pu.c(2));
else
    fprintf(fid, 'Pu = [%.15g; %.15g]\n', Pu(1), Pu(2));
end
fprintf(fid, 'Pu_true = [%.15g; %.15g]\n', Pu_true(1), Pu_true(2));
fprintf(fid, 'Max difference = %.15g\n', max(abs(diff)));
fprintf(fid, 'compareMatrices passed: %d\n', compareMatrices(Pu,Pu_true,tol,'equal',true));
fclose(fid);

fprintf('\nResults saved to affineSolution_matlab_output.txt\n');
