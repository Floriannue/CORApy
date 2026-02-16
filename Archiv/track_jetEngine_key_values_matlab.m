% track_jetEngine_key_values_matlab - Track varphi, finitehorizon, and timeStep values in MATLAB

addpath('cora_matlab/models/auxiliary/jetEngine');

dim_x = 2;
params.tFinal = 8;
params.R0 = zonotope([[1;1],0.1*diag(ones(dim_x,1))]);
params.U = zonotope(0);

options.alg = 'lin-adaptive';

sys = nonlinearSys(@jetEngine,dim_x,1);

fprintf('Tracking key values in MATLAB...\n');
[R,~,opt] = reach(sys,params,options);

fprintf('\n=== KEY VALUES ANALYSIS ===\n');

if isfield(opt, 'finitehorizon') && length(opt.finitehorizon) > 0
    fprintf('\nFirst 20 finitehorizon values:\n');
    for i = 1:min(20, length(opt.finitehorizon))
        fprintf('  Step %d: finitehorizon = %.6e\n', i+1, opt.finitehorizon(i));
    end
end

if isfield(opt, 'varphi') && length(opt.varphi) > 0
    fprintf('\nFirst 20 varphi values:\n');
    for i = 1:min(20, length(opt.varphi))
        fprintf('  Step %d: varphi = %.6f\n', i+1, opt.varphi(i));
    end
    fprintf('\n  Min varphi: %.6f\n', min(opt.varphi(opt.varphi > 0)));
    fprintf('  Max varphi: %.6f\n', max(opt.varphi));
    fprintf('  Mean varphi: %.6f\n', mean(opt.varphi(opt.varphi > 0)));
end

if isfield(opt, 'stepsize') && length(opt.stepsize) > 0
    fprintf('\nFirst 20 stepsize values:\n');
    for i = 1:min(20, length(opt.stepsize))
        fprintf('  Step %d: stepsize = %.6e\n', i+1, opt.stepsize(i));
    end
end

% Check finitehorizon growth
fprintf('\n=== FINITEHORIZON GROWTH ANALYSIS ===\n');
if isfield(opt, 'finitehorizon') && length(opt.finitehorizon) > 1
    for i = 2:min(20, length(opt.finitehorizon))
        prev_fh = opt.finitehorizon(i-1);
        curr_fh = opt.finitehorizon(i);
        if isfield(opt, 'varphi') && length(opt.varphi) >= i-1
            prev_varphi = opt.varphi(i-1);
            minorder = opt.minorder;
            zetaphi_val = opt.zetaphi(minorder+1);  % MATLAB is 1-indexed
            computed = prev_fh * (1 + prev_varphi - zetaphi_val);
            remTime = params.tFinal - (i * mean(opt.stepsize(1:i-1)));  % Approximate
            fprintf('Step %d: prev_fh=%.6e, varphi=%.6f, zetaphi=%.6f, computed=%.6e, remTime≈%.6f\n', ...
                i+1, prev_fh, prev_varphi, zetaphi_val, computed, remTime);
        end
    end
end
