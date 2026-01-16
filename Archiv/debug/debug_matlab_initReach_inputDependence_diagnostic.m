% Diagnostic script to understand the iPow issue
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Diagnostic: Understanding iPow structure ===\n\n');

% Setup with parametric A
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];
A = matZonotope(A_center, A_gen);
B = [0; 1];
c = [0; 0];
sys = linParamSys(A, B, c, 'constParam');

% Parameters
Uconst = zonotope([0; 0], [0.05, 0; 0, 0.05]);
U = zonotope(0, 0.05);
uTrans = 0.1;
params.Uconst = Uconst;
params.U = U;
params.uTrans = uTrans;

% Options
options.timeStep = 0.1;
options.taylorTerms = 4;
options.reductionTechnique = 'girard';
options.zonotopeOrder = 10;
options.compTimePoint = true;
options.intermediateTerms = 2;
options.originContained = false;

fprintf('Configuration:\n');
fprintf('  intermediateTerms = %d\n', options.intermediateTerms);
fprintf('  taylorTerms = %d\n', options.taylorTerms);
fprintf('  Loop in priv_highOrderMappingMatrix: i = %d:%d\n', ...
    options.intermediateTerms+1, options.taylorTerms);
fprintf('\n');

% Set stepSize and taylorTerms (these are set by initReach_inputDependence)
% But we need to set them for priv_mappingMatrix to work
% Actually, let's just call initReach_inputDependence and see where it fails
fprintf('Calling initReach_inputDependence (which sets stepSize and taylorTerms)...\n');
try
    [sys_out, Rfirst, options_out] = initReach_inputDependence(sys, zonotope([0;0], [0.1,0;0,0.1]), params, options);
    fprintf('  Success! initReach_inputDependence completed\n');
catch ME
    fprintf('  Error in initReach_inputDependence: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
    
    % Try to manually set properties and call priv_mappingMatrix
    fprintf('Trying manual approach...\n');
    % We can't set stepSize directly, so let's try a different approach
    % Actually, let's check if we can access the private setter or use a workaround
    return;
end

% If we got here, inspect the result
fprintf('\nInspecting sys.power.int after initReach_inputDependence:\n');
if isfield(sys_out, 'power') && isfield(sys_out.power, 'int')
    fprintf('  obj.power.int type: %s\n', class(sys_out.power.int));
    fprintf('  obj.power.int is cell: %d\n', iscell(sys_out.power.int));
    if iscell(sys_out.power.int)
        fprintf('  obj.power.int length: %d\n', length(sys_out.power.int));
        fprintf('  obj.power.int non-empty indices: ');
        for i=1:length(sys_out.power.int)
            if ~isempty(sys_out.power.int{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    end
end

% Now try calling priv_mappingMatrix directly to inspect
fprintf('\nTrying to call priv_mappingMatrix directly...\n');
% We need to set stepSize and taylorTerms, but they're read-only
% Let's use a workaround - create a new system and call the function
% Actually, let's just inspect what we have
return;

% OLD CODE - trying priv_mappingMatrix directly
fprintf('Calling priv_mappingMatrix...\n');
try
    sys = priv_mappingMatrix(sys, params, options);
    fprintf('  Success!\n');
    fprintf('  obj.power.int type: %s\n', class(sys.power.int));
    fprintf('  obj.power.int is cell: %d\n', iscell(sys.power.int));
    if iscell(sys.power.int)
        fprintf('  obj.power.int length: %d\n', length(sys.power.int));
        fprintf('  obj.power.int indices: ');
        for i=1:length(sys.power.int)
            if ~isempty(sys.power.int{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    end
catch ME
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end

fprintf('\n');

% Try to access what priv_highOrderMappingMatrix would access
if iscell(sys.power.int)
    fprintf('Testing access to iPow elements:\n');
    for i=(options.intermediateTerms+1):options.taylorTerms
        fprintf('  Trying to access iPow{%d}...', i);
        try
            if i <= length(sys.power.int) && ~isempty(sys.power.int{i})
                fprintf(' Success (exists)\n');
            else
                fprintf(' Empty or out of bounds\n');
            end
        catch ME
            fprintf(' Error: %s\n', ME.message);
        end
    end
end

fprintf('\n');

% Try calling priv_highOrderMappingMatrix
fprintf('Calling priv_highOrderMappingMatrix...\n');
try
    sys = priv_highOrderMappingMatrix(sys, options.intermediateTerms);
    fprintf('  Success!\n');
catch ME
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
