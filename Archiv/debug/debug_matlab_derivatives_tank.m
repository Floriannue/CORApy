% Debug script: generate derivatives for tank6Eq and verify files

clear; clc;
tank = nonlinearSys(@tank6Eq);
fprintf('sys.name = %s\n', tank.name);

options.tensorOrder = 2;
options.tensorOrderOutput = 2;
options.verbose = true;

derivatives(tank, options);

path = [CORAROOT filesep 'models' filesep 'auxiliary' filesep tank.name];
fprintf('auxiliary path: %s\n', path);
addpath(path);
if exist(path, 'dir')
    d = dir(path);
    for i=1:length(d)
        fprintf('file: %s\n', d(i).name);
    end
end

fprintf('jacobian exists: %d\n', exist('jacobian_tank6Eq', 'file'));
fprintf('out_jacobian exists: %d\n', exist('out_jacobian_tank6Eq', 'file'));
