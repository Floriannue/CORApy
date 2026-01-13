% Test what expmMixed returns
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Testing expmMixed return values ===\n\n');

% Create a simple matZonotope (like our test)
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];
A = matZonotope(A_center, A_gen);

% Parameters
r = 0.1;
intermediateTerms = 2;
maxOrder = 4;

% Multiply by stepSize (as done in priv_mappingMatrix)
A_scaled = A * r;

fprintf('Calling expmMixed with:\n');
fprintf('  A: matZonotope %dx%d, %d generators\n', size(A_scaled.C,1), size(A_scaled.C,2), A_scaled.numgens);
fprintf('  r = %.15g\n', r);
fprintf('  intermediateTerms = %d\n', intermediateTerms);
fprintf('  maxOrder = %d\n', maxOrder);
fprintf('\n');

try
    [eZ, eI, zPow, iPow, E] = expmMixed(A_scaled, r, intermediateTerms, maxOrder);
    
    fprintf('Return values:\n');
    fprintf('  eZ type: %s\n', class(eZ));
    fprintf('  eI type: %s\n', class(eI));
    fprintf('  zPow type: %s, is cell: %d\n', class(zPow), iscell(zPow));
    if iscell(zPow)
        fprintf('    zPow length: %d\n', length(zPow));
        fprintf('    zPow non-empty: ');
        for i=1:length(zPow)
            if ~isempty(zPow{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    end
    fprintf('  iPow type: %s, is cell: %d\n', class(iPow), iscell(iPow));
    if iscell(iPow)
        fprintf('    iPow length: %d\n', length(iPow));
        fprintf('    iPow non-empty: ');
        for i=1:length(iPow)
            if ~isempty(iPow{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    else
        fprintf('    ERROR: iPow is NOT a cell array!\n');
        fprintf('    iPow value: %s\n', mat2str(iPow));
    end
    fprintf('  E type: %s\n', class(E));
    
    % Now check what happens when we try to store it
    fprintf('\nTesting storage in obj.power:\n');
    obj.power = [];
    fprintf('  obj.power before: %s\n', class(obj.power));
    
    obj.power.zono = zPow;
    fprintf('  obj.power.zono after setting: type %s, is cell: %d\n', class(obj.power.zono), iscell(obj.power.zono));
    
    obj.power.int = iPow;
    fprintf('  obj.power.int after setting: type %s, is cell: %d\n', class(obj.power.int), iscell(obj.power.int));
    
    % Now test accessing it
    fprintf('\nTesting access (as done in priv_highOrderMappingMatrix):\n');
    for i=(intermediateTerms+1):maxOrder
        fprintf('  obj.power.int{%d}: ', i);
        if iscell(obj.power.int) && i <= length(obj.power.int) && ~isempty(obj.power.int{i})
            fprintf('OK, type: %s\n', class(obj.power.int{i}));
        else
            fprintf('ERROR - cannot access\n');
        end
    end
    
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
