% Debug script to check what priv_expmRemainder returns
clear; close all; clc;

% Create a simple 2D system
A = [-1, 2; -2, -1];
linsys = linearSys(A);

% Initialize taylor object
linsys.taylor = taylorLinSys(A);

timeStep = 0.1;
truncationOrder = 10;

% Call priv_expmRemainder
E = priv_expmRemainder(linsys, timeStep, truncationOrder);

fprintf('=== priv_expmRemainder Results ===\n');
fprintf('E type: %s\n', class(E));
if isa(E, 'intervalMatrix')
    fprintf('E.inf shape: %s\n', mat2str(size(E.int.inf)));
    fprintf('E.sup shape: %s\n', mat2str(size(E.int.sup)));
    fprintf('E.inf:\n');
    disp(E.int.inf);
    fprintf('E.sup:\n');
    disp(E.int.sup);
    fprintf('E.infimum():\n');
    disp(infimum(E));
    fprintf('E.supremum():\n');
    disp(supremum(E));
    fprintf('Check: E.infimum() == -E.supremum()?\n');
    fprintf('  Result: %d\n', isequal(infimum(E), -supremum(E)));
    fprintf('  Max difference: %e\n', max(abs(infimum(E) - (-supremum(E))), [], 'all'));
elseif isa(E, 'interval')
    fprintf('E.inf shape: %s\n', mat2str(size(E.inf)));
    fprintf('E.sup shape: %s\n', mat2str(size(E.sup)));
    fprintf('E.inf:\n');
    disp(E.inf);
    fprintf('E.sup:\n');
    disp(E.sup);
end
