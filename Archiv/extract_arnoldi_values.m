% Quick script to extract exact A and vInit values
rng(42);
A = randn(5, 5);
A = (A + A.') / 2;  % Make symmetric

vInit = randn(5, 1);
vInit = vInit / norm(vInit);  % Normalize

% Print A matrix
fprintf('A = np.array([\n');
for i = 1:5
    fprintf('    [');
    for j = 1:5
        fprintf('%.15g', A(i,j));
        if j < 5
            fprintf(', ');
        end
    end
    fprintf(']');
    if i < 5
        fprintf(',\n');
    else
        fprintf('\n');
    end
end
fprintf('])\n\n');

% Print vInit
fprintf('vInit = np.array([[');
for i = 1:5
    fprintf('%.15g', vInit(i));
    if i < 5
        fprintf('], [');
    end
end
fprintf(']])\n');
