% Debug script to add debugging to priv_highOrderMappingMatrix
% This will help us understand what obj.power.int actually is

% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Debugging priv_highOrderMappingMatrix ===\n\n');

% Create a simple matZonotope
A_center = [0, 1; -1, -0.5];
A_gen = zeros(2, 2, 1);
A_gen(:,:,1) = [0.1, 0; 0, 0.1];
A = matZonotope(A_center, A_gen);
B = [0; 1];
c = [0; 0];
sys = linParamSys(A, B, c, 'constParam');

% Initial set
Rinit = zonotope([0; 0], [0.1, 0; 0, 0.1]);

% Parameters
Uconst = zonotope([0; 0], [0.05, 0; 0, 0.05]);
U = zonotope([0; 0], [0.05, 0; 0, 0.05]);
uTrans = [0.1; 0];
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

% Call initReach_inputDependence but stop before priv_highOrderMappingMatrix
fprintf('Step 1: Setting stepSize and taylorTerms...\n');
% We can't set these directly, so we need to call initReach_inputDependence
% But let's try to manually call priv_mappingMatrix first to see what happens

% Actually, let's create a modified version that adds debugging
% Or better yet, let's check what happens if we manually trace through

fprintf('Step 2: Calling priv_mappingMatrix manually...\n');
% We need stepSize and taylorTerms set first
% Since we can't set them, let's check if we can work around this
% Actually, let's just call initReach_inputDependence and see where it fails

% Let's add a breakpoint by creating a wrapper
fprintf('Calling initReach_inputDependence...\n');
try
    % Save original priv_highOrderMappingMatrix
    original_priv_highOrderMappingMatrix = @priv_highOrderMappingMatrix;
    
    % Create a wrapper that adds debugging
    function obj_out = debug_priv_highOrderMappingMatrix(obj, intermediateTerms)
        fprintf('  DEBUG: Inside priv_highOrderMappingMatrix\n');
        fprintf('    obj.power type: %s\n', class(obj.power));
        fprintf('    obj.power.int type: %s\n', class(obj.power.int));
        fprintf('    obj.power.int is cell: %d\n', iscell(obj.power.int));
        if iscell(obj.power.int)
            fprintf('    obj.power.int length: %d\n', length(obj.power.int));
        end
        fprintf('    intermediateTerms: %d\n', intermediateTerms);
        fprintf('    obj.taylorTerms: %d\n', obj.taylorTerms);
        fprintf('    Loop range: %d:%d\n', intermediateTerms+1, obj.taylorTerms);
        
        % Call original
        obj_out = original_priv_highOrderMappingMatrix(obj, intermediateTerms);
    end
    
    % Replace the function temporarily (this won't work in MATLAB)
    % Instead, let's just call and catch the error with more info
    
    [sys_out, Rfirst, options_out] = initReach_inputDependence(sys, Rinit, params, options);
    fprintf('  Success!\n');
    
catch ME
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    
    % Try to inspect obj.power.int at the point of error
    if strcmp(ME.stack(1).name, 'priv_highOrderMappingMatrix')
        fprintf('\n  Attempting to inspect obj.power.int structure...\n');
        % We can't access the local variables, but we can try to recreate the state
        fprintf('  This suggests obj.power.int is not a cell array when accessed\n');
        fprintf('  Possible causes:\n');
        fprintf('    1. obj.power.int was never set (empty)\n');
        fprintf('    2. obj.power.int was overwritten with non-cell value\n');
        fprintf('    3. obj.power is not a struct\n');
    end
end
