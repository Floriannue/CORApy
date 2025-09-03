% Debug script for splitLongestGen - compare with Python
clear; clc;

% Test the same case as Python
c = [-1; 3];
G = [2, 0, 1; 1, 2, 1];
E = [1, 0, 2; 0, 1, 1];
GI = zeros(2, 0);
pZ = polyZonotope(c, G, GI, E);

fprintf('MATLAB Original pZ:\n');
fprintf('  c: [%.1f, %.1f]\n', pZ.c(1), pZ.c(2));
fprintf('  G shape: %dx%d\n', size(pZ.G, 1), size(pZ.G, 2));
fprintf('  E shape: %dx%d\n', size(pZ.E, 1), size(pZ.E, 2));

% Show generator lengths
len_gen = sum(pZ.G.^2, 1);
fprintf('  Generator lengths: [%.1f, %.1f, %.1f]\n', len_gen(1), len_gen(2), len_gen(3));
fprintf('  Longest generator index: %d\n', find(len_gen == max(len_gen)));

% Show which factor has largest exponent in longest generator
[~, ind] = max(len_gen);
fprintf('  Longest generator column: %d\n', ind);
fprintf('  E(:, %d): [%d, %d]\n', ind, pZ.E(1, ind), pZ.E(2, ind));
[~, factor_idx] = max(pZ.E(:, ind));
fprintf('  Factor with largest exponent index: %d\n', factor_idx);
fprintf('  Factor value: %d\n', pZ.id(factor_idx));

pZsplit = splitLongestGen(pZ);
fprintf('\nSplit into %d parts:\n', length(pZsplit));
for i = 1:length(pZsplit)
    fprintf('  pZsplit{%d}:\n', i);
    fprintf('    c: [%.1f, %.1f]\n', pZsplit{i}.c(1), pZsplit{i}.c(2));
    fprintf('    G shape: %dx%d\n', size(pZsplit{i}.G, 1), size(pZsplit{i}.G, 2));
    fprintf('    E shape: %dx%d\n', size(pZsplit{i}.E, 1), size(pZsplit{i}.E, 2));
    fprintf('    G:\n');
    disp(pZsplit{i}.G);
    fprintf('    E:\n');
    disp(pZsplit{i}.E);
end

fprintf('\nMATLAB splitLongestGen test completed successfully!\n');
