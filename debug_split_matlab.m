% Debug script to understand MATLAB splitLongestGen behavior
clear; clc;

% Create a simple polynomial zonotope (same as Python test)
c = [0; 0];
G = [2 0 1; 0 2 1];
GI = [0; 0];
E = [1 0 3; 0 1 1];
id = [1; 2];

pZ = polyZonotope(c, G, GI, E, id);

fprintf('MATLAB PolyZonotope:\n');
fprintf('  c: [%g; %g]\n', c(1), c(2));
fprintf('  G shape: %dx%d\n', size(G,1), size(G,2));
fprintf('  E shape: %dx%d\n', size(E,1), size(E,2));
fprintf('  id: [%g; %g]\n', id(1), id(2));

% Test splitLongestGen
fprintf('\nTesting splitLongestGen...\n');

% Determine longest generator
len = sum(pZ.G.^2,1);
fprintf('  Generator lengths: [%g, %g, %g]\n', len(1), len(2), len(3));
[~,ind] = max(len);
fprintf('  Longest generator index: %d\n', ind);

% Find factor with the largest exponent
fprintf('  E(:, %d): [%g; %g]\n', ind, pZ.E(1,ind), pZ.E(2,ind));
[~,factor_idx] = max(pZ.E(:,ind));
fprintf('  Factor with largest exponent index: %d\n', factor_idx);
factor = pZ.id(factor_idx);
fprintf('  Factor value: %g\n', factor);

% Test splitDepFactor
fprintf('\nTesting splitDepFactor with factor %g...\n', factor);

% Find selected dependent factor
ind_mask = pZ.id == factor;
fprintf('  pZ.id == %g: [%d; %d]\n', factor, ind_mask(1), ind_mask(2));
fprintf('  sum(ind_mask): %d\n', sum(ind_mask));

E_ind = pZ.E(ind_mask, :);
fprintf('  E_ind shape: %dx%d\n', size(E_ind,1), size(E_ind,2));
fprintf('  E_ind: [%g, %g, %g]\n', E_ind(1,1), E_ind(1,2), E_ind(1,3));

% Parse input arguments
polyOrd = max(E_ind);
fprintf('  polyOrd: %d\n', polyOrd);

% Determine all generators in which the selected dependent factor occurs
genInd = 0 < E_ind & E_ind <= polyOrd;
fprintf('  genInd: [%d, %d, %d]\n', genInd(1), genInd(2), genInd(3));

fprintf('\nMATLAB splitLongestGen test completed successfully!\n');
