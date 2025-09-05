% Debug script to trace center calculation in splitDepFactor
clear; clc;

% Test the same case as Python
c = [-1; 3];
G = [2, 0, 1; 1, 2, 1];
E = [1, 0, 2; 0, 1, 1];
GI = zeros(2, 0);
pZ = polyZonotope(c, G, GI, E);

fprintf('Original pZ:\n');
fprintf('  c: [%.1f, %.1f]\n', pZ.c(1), pZ.c(2));
fprintf('  G:\n');
disp(pZ.G);
fprintf('  E:\n');
disp(pZ.E);

% Manually trace through splitDepFactor logic
ind = 1;  % factor to split
polyOrd = 2;

% Find selected dependent factor
ind_mask = (pZ.id == ind);
E_ind = E(ind_mask, :);
fprintf('\nE_ind: [%d, %d, %d]\n', E_ind(1), E_ind(2), E_ind(3));

% Determine generators to split
genInd = (0 < E_ind & E_ind <= polyOrd);
fprintf('genInd: [%d, %d, %d]\n', genInd(1), genInd(2), genInd(3));

% Create polynomial coefficients
polyCoeff1 = cell(max(2,polyOrd),1);
polyCoeff2 = cell(max(2,polyOrd),1);

% Create pascal triangle
P = {[1, 1]};
for i=2:polyOrd
    P{i} = [1 sum(P{i-1}([1:(i-1); 2:i])) 1];
end

fprintf('Pascal triangle P: ');
for i=1:length(P)
    fprintf('[%s] ', num2str(P{i}));
end
fprintf('\n');

for i=1:polyOrd
    Pi = P{i};
    polyCoeff1{i} = 0.5^i * Pi;
    polyCoeff2{i} = 0.5^i * Pi .* (-mod(i:-1:0, 2)*2+1);
    fprintf('i=%d: Pi=[%s]\n', i, num2str(Pi));
    fprintf('  polyCoeff1{%d} = [%s]\n', i, num2str(polyCoeff1{i}));
    fprintf('  polyCoeff2{%d} = [%s]\n', i, num2str(polyCoeff2{i}));
end

% Initialize centers
c1 = pZ.c;
c2 = pZ.c;
fprintf('\nInitial centers: c1=[%.1f, %.1f], c2=[%.1f, %.1f]\n', c1(1), c1(2), c2(1), c2(2));

% Calculate hout
hout = sum(~genInd);
for i=1:polyOrd
    numExpi = sum(E_ind == i);
    hout = hout + length(P{i}) * numExpi;
end

fprintf('hout: %d\n', hout);

% Create output matrices
G1 = nan(length(c1), hout);
G2 = nan(length(c2), hout);
Eout = nan(size(E, 1), hout);

% Fill in non-split generators
h = 1;
dh = sum(~genInd);
if dh > 0
    G1(:, h:h+dh-1) = pZ.G(:, ~genInd);
    G2(:, h:h+dh-1) = pZ.G(:, ~genInd);
    Eout(:, h:h+dh-1) = E(:, ~genInd);
    h = h + dh;
end

fprintf('After non-split generators: h=%d\n', h);
fprintf('G1(:, 1:%d):\n', h-1);
disp(G1(:, 1:h-1));
fprintf('G2(:, 1:%d):\n', h-1);
disp(G2(:, 1:h-1));

% Process each polynomial order
for i=1:polyOrd
    coef1 = polyCoeff1{i};
    coef2 = polyCoeff2{i};
    
    expi = (E_ind == i);
    dh = length(coef1) * sum(expi);
    
    fprintf('\ni=%d: expi=[%d, %d, %d], dh=%d\n', i, expi(1), expi(2), expi(3), dh);
    fprintf('coef1=[%s], coef2=[%s]\n', num2str(coef1), num2str(coef2));
    
    if sum(expi) > 0
        G1(:, h:h+dh-1) = kron(coef1, pZ.G(:, expi));
        G2(:, h:h+dh-1) = kron(coef2, pZ.G(:, expi));
        
        fprintf('G1(:, %d:%d):\n', h, h+dh-1);
        disp(G1(:, h:h+dh-1));
        fprintf('G2(:, %d:%d):\n', h, h+dh-1);
        disp(G2(:, h:h+dh-1));
        
        % Repeat E columns
        Eout(:, h:h+dh-1) = repmat(E(:, expi),1,i+1);
        Eout(ind, h:h+dh-1) = kron(0:i, ones(1, sum(expi)));
        
        h = h + dh;
        genInd = xor(genInd, expi);
    end
end

fprintf('\nFinal G1:\n');
disp(G1);
fprintf('Final G2:\n');
disp(G2);
fprintf('Final Eout:\n');
disp(Eout);

% Add generators with all-zero exponent matrix to center
temp = sum(Eout,1);
genInd_zero = (temp == 0);
fprintf('\ngenInd_zero: [%s]\n', num2str(genInd_zero));
fprintf('G1(:, genInd_zero):\n');
disp(G1(:,genInd_zero));
fprintf('G2(:, genInd_zero):\n');
disp(G2(:,genInd_zero));

c1_final = c1 + sum(G1(:,genInd_zero),2);
c2_final = c2 + sum(G2(:,genInd_zero),2);

fprintf('\nFinal centers:\n');
fprintf('c1_final: [%.1f, %.1f]\n', c1_final(1), c1_final(2));
fprintf('c2_final: [%.1f, %.1f]\n', c2_final(1), c2_final(2));

fprintf('\nMATLAB center calculation test completed!\n');
