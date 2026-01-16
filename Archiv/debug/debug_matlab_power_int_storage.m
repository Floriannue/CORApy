% Debug script to understand how obj.power.int is stored
% Add CORA to path if needed
if ~exist('zonotope', 'file')
    addpath(genpath('cora_matlab'));
end

fprintf('=== Investigating obj.power.int storage ===\n\n');

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

% Parameters - based on linearize.m analysis:
% After linearization, Uconst and uTrans are in state space dimension
% (because B=1 after linearization)
Uconst = zonotope([0; 0], [0.05, 0; 0, 0.05]);  % State space dimension
U = zonotope([0; 0], [0.05, 0; 0, 0.05]);  % State space dimension (after linearization)
uTrans = [0.1; 0];  % State space dimension
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

% Set stepSize and taylorTerms (normally done by initReach_inputDependence)
% But we can't set them directly, so we'll call priv_mappingMatrix which sets them
% Actually, let's manually set them using a workaround
fprintf('Setting sys properties...\n');
% We need to set these, but they're read-only. Let's try calling priv_mappingMatrix
% which might set them internally, or we need to find another way.

% Actually, let's trace through what happens step by step
% Since stepSize and taylorTerms are SetAccess=private, we need to call
% initReach_inputDependence which is a class method and can set them
fprintf('Calling initReach_inputDependence (which sets stepSize and taylorTerms)...\n');
try
    [sys_out, Rfirst, options_out] = initReach_inputDependence(sys, Rinit, params, options);
    
    fprintf('  Success! initReach_inputDependence completed\n');
    fprintf('  sys_out.power.int type: %s\n', class(sys_out.power.int));
    fprintf('  sys_out.power.int is cell: %d\n', iscell(sys_out.power.int));
    if iscell(sys_out.power.int)
        fprintf('  sys_out.power.int length: %d\n', length(sys_out.power.int));
        fprintf('  sys_out.power.int non-empty indices: ');
        for i=1:length(sys_out.power.int)
            if ~isempty(sys_out.power.int{i})
                fprintf('%d ', i);
            end
        end
        fprintf('\n');
    else
        fprintf('  ERROR: sys_out.power.int is NOT a cell array!\n');
        fprintf('  sys_out.power.int value/type: %s\n', class(sys_out.power.int));
        if isnumeric(sys_out.power.int)
            fprintf('  sys_out.power.int size: %s\n', mat2str(size(sys_out.power.int)));
        end
    end
    
    fprintf('\n  Now checking what priv_highOrderMappingMatrix would access:\n');
    for i=(options.intermediateTerms+1):sys_out.taylorTerms
        fprintf('    sys_out.power.int{%d}: ', i);
        if iscell(sys_out.power.int) && i <= length(sys_out.power.int) && ~isempty(sys_out.power.int{i})
            fprintf('exists, type: %s\n', class(sys_out.power.int{i}));
        else
            fprintf('DOES NOT EXIST or EMPTY\n');
        end
    end
    
    fprintf('\n  Results:\n');
    fprintf('    Rfirst.tp center shape: %s\n', mat2str(size(center(Rfirst.tp))));
    fprintf('    Rfirst.ti center shape: %s\n', mat2str(size(center(Rfirst.ti))));
    
catch ME
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Stack:\n');
    for i=1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
