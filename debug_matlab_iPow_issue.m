% Debug script to understand the iPow issue in priv_highOrderMappingMatrix
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Investigating iPow structure from expm ===\n\n');

% Create a simple matZonotope
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];
A = matZonotope(A_center, A_gen);

% Parameters matching our test
r = 0.1;
intermediateOrder = 2;
maxOrder = 4;

fprintf('Parameters:\n');
fprintf('  r = %.15g\n', r);
fprintf('  intermediateOrder = %d\n', intermediateOrder);
fprintf('  maxOrder = %d\n', maxOrder);
fprintf('  initialOrder (for expm) = %d\n', intermediateOrder+1);
fprintf('\n');

% Multiply by stepSize (as done in priv_mappingMatrix)
A_scaled = A * r;

% Call expmMixed (as done in priv_mappingMatrix)
fprintf('Calling expmMixed...\n');
try
    [eZ, eI, zPow, iPow, E] = expmMixed(A_scaled, r, intermediateOrder, maxOrder);
    fprintf('  Success!\n');
    fprintf('  iPow type: %s\n', class(iPow));
    fprintf('  iPow is cell: %d\n', iscell(iPow));
    if iscell(iPow)
        fprintf('  iPow length: %d\n', length(iPow));
        fprintf('  iPow non-empty indices: ');
        for i=1:length(iPow)
            if ~isempty(iPow{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
        
        % Check what indices are accessed in priv_highOrderMappingMatrix
        fprintf('\n  Indices that priv_highOrderMappingMatrix would access:\n');
        for i=(intermediateOrder+1):maxOrder
            fprintf('    iPow{%d}: ', i);
            if i <= length(iPow) && ~isempty(iPow{i})
                fprintf('exists, type: %s\n', class(iPow{i}));
            else
                fprintf('DOES NOT EXIST or EMPTY\n');
            end
        end
    else
        fprintf('  ERROR: iPow is NOT a cell array!\n');
        fprintf('  iPow value: %s\n', mat2str(iPow));
    end
catch ME
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

fprintf('\n');

% Now check what expm returns directly
fprintf('Calling expm directly with same parameters...\n');
intMat = intervalMatrix(A_scaled);
initialOrder = intermediateOrder + 1;
zPow_intermediate = zPow{intermediateOrder};
initialPower = intMat * intervalMatrix(zPow_intermediate);

try
    [eI_direct, iPow_direct, E_direct] = expm(intMat, r, maxOrder, initialOrder, initialPower);
    fprintf('  Success!\n');
    fprintf('  iPow_direct type: %s\n', class(iPow_direct));
    fprintf('  iPow_direct is cell: %d\n', iscell(iPow_direct));
    if iscell(iPow_direct)
        fprintf('  iPow_direct length: %d\n', length(iPow_direct));
        fprintf('  iPow_direct non-empty indices: ');
        for i=1:length(iPow_direct)
            if ~isempty(iPow_direct{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    end
catch ME
    fprintf('  Error: %s\n', ME.message);
end
