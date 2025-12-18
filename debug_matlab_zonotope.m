% Debug script to check MATLAB behavior
c = [2;-1];
G = [1;-1];
GI = [2 0 1; -1 1 0];

% Test case 1: Only independent generators
pZ1 = polyZonotope(c,[],GI);
[res1,Z1] = representsa(pZ1,'zonotope');
fprintf('Test 1 - Only GI: res=%d\n', res1);

% Test case 2: Dependent and independent generators (no E specified)
pZ2 = polyZonotope(c,G,GI);
[res2,Z2] = representsa(pZ2,'zonotope');
fprintf('Test 2 - G and GI (no E): res=%d\n', res2);

% Test case 3: With exponent matrix
E = [3; 0];
pZ3 = polyZonotope(c,G,GI,E);
fprintf('pZ3.E shape: [%d, %d]\n', size(pZ3.E,1), size(pZ3.E,2));
fprintf('pZ3.G shape: [%d, %d]\n', size(pZ3.G,1), size(pZ3.G,2));
fprintf('pZ3.E:\n');
disp(pZ3.E);
fprintf('pZ3.G:\n');
disp(pZ3.G);

% Check aux_isZonotope
res_aux = aux_isZonotope(pZ3, 1e-10);
fprintf('aux_isZonotope result: %d\n', res_aux);

[res3,Z3] = representsa(pZ3,'zonotope');
fprintf('Test 3 - With E=[3;0]: res=%d\n', res3);
fprintf('Z3 center:\n');
disp(Z3.c);
fprintf('Z3 generators:\n');
disp(Z3.G);

