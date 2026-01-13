% Detailed MATLAB debug script for initReach
% Matches Python investigation step-by-step

% Setup
dim_x = 6;
params.R0 = zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(dim_x));
params.U = zonotope(zeros(1,1), 0.005*eye(1));
params.tFinal = 4;
params.uTrans = zeros(1,1);

options.timeStep = 4;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;
options.maxError = inf(dim_x,1);

% System
tank = nonlinearSys(@tank6Eq,6,1);

% Validate options and compute derivatives
[params,options] = validateOptions(tank,params,options,'FunctionName','reach');
derivatives(tank,options);

fprintf('================================================================================\n');
fprintf('DETAILED PRECISION LOSS INVESTIGATION (MATLAB)\n');
fprintf('================================================================================\n');

% Step 1: Initial set
fprintf('\n[STEP 1] Initial Set R0\n');
fprintf('  R0 center: [%s]\n', sprintf('%.15e ', params.R0.c));
fprintf('  R0 generators shape: %dx%d\n', size(params.R0.G));
if size(params.R0.G,2) > 0
    fprintf('  R0 first generator: [%s]\n', sprintf('%.15e ', params.R0.G(:,1)));
end

% Step 2: Linearization
fprintf('\n[STEP 2] Linearization\n');
[sys, linsys, linParams, linOptions] = linearize(tank, params.R0, params, options);
fprintf('  Linearization point p.x: [%s]\n', sprintf('%.15e ', sys.linError.p.x));
fprintf('  Linearization point p.u: [%s]\n', sprintf('%.15e ', sys.linError.p.u));
fprintf('  f0 (constant input): [%s]\n', sprintf('%.15e ', sys.linError.f0));
fprintf('  Jacobian A shape: %dx%d\n', size(linsys.A));
fprintf('  Jacobian A(1, :): [%s]\n', sprintf('%.15e ', linsys.A(1,:)));
if isfield(linsys, 'B')
    fprintf('  Jacobian B shape: %dx%d\n', size(linsys.B));
end

% Step 3: Translate Rinit
fprintf('\n[STEP 3] Translate Rinit\n');
Rdelta = params.R0 + (-sys.linError.p.x);
fprintf('  Rdelta center: [%s]\n', sprintf('%.15e ', Rdelta.c));
fprintf('  Rdelta generators shape: %dx%d\n', size(Rdelta.G));
fprintf('  Rdelta center first 3: [%s]\n', sprintf('%.15e ', Rdelta.c(1:3)));

% Step 4: Compute eAt
fprintf('\n[STEP 4] Matrix Exponential eAt\n');
A = linsys.A;
timeStep = options.timeStep;
eAt = expm(A * timeStep);
fprintf('  eAt shape: %dx%d\n', size(eAt));
fprintf('  eAt(1,1): %.15e\n', eAt(1,1));
fprintf('  eAt(1,2): %.15e\n', eAt(1,2));
fprintf('  eAt max value: %.15e\n', max(eAt(:)));
fprintf('  eAt min value: %.15e\n', min(eAt(:)));

% Step 5: Homogeneous solution
fprintf('\n[STEP 5] Homogeneous Solution\n');
[Htp, Hti, C_state] = linsys.homogeneousSolution(Rdelta, timeStep, options.taylorTerms);
fprintf('  Htp center: [%s]\n', sprintf('%.15e ', Htp.c));
fprintf('  Htp generators shape: %dx%d\n', size(Htp.G));
fprintf('  Htp center first 3: [%s]\n', sprintf('%.15e ', Htp.c(1:3)));

% Check eAt * Rdelta manually
fprintf('\n[STEP 5a] Manual eAt * Rdelta check\n');
eAt_c = eAt * Rdelta.c;
eAt_G = eAt * Rdelta.G;
fprintf('  eAt * Rdelta.c: [%s]\n', sprintf('%.15e ', eAt_c(1:3)));
fprintf('  eAt * Rdelta.G shape: %dx%d\n', size(eAt_G));
if size(eAt_G,2) > 0
    fprintf('  eAt * Rdelta.G first column: [%s]\n', sprintf('%.15e ', eAt_G(:,1)));
end
diff = Htp.c - eAt_c;
fprintf('  Htp.c matches eAt * Rdelta.c? %s\n', mat2str(all(abs(diff) < 1e-14)));
if any(abs(diff) >= 1e-14)
    fprintf('  Difference: [%s]\n', sprintf('%.15e ', diff(1:3)));
    fprintf('  Max abs diff: %.15e\n', max(abs(diff)));
end

% Step 6: Particular solutions
fprintf('\n[STEP 6] Particular Solutions\n');
[Pu, C_input_const, ~] = linsys.particularSolution_constant(linParams.uTrans, timeStep, options.taylorTerms);
PU = linsys.particularSolution_timeVarying(linParams.U, timeStep, options.taylorTerms);
if isa(Pu, 'zonotope')
    fprintf('  Pu center: [%s]\n', sprintf('%.15e ', Pu.c));
else
    fprintf('  Pu (numeric): [%s]\n', sprintf('%.15e ', Pu));
end
fprintf('  PU center: [%s]\n', sprintf('%.15e ', PU.c));
fprintf('  PU generators shape: %dx%d\n', size(PU.G));

% Step 7: Zonotope addition
fprintf('\n[STEP 7] Zonotope Addition (Rtp = Htp + PU + Pu)\n');
fprintf('  Before addition:\n');
fprintf('    Htp center: [%s]\n', sprintf('%.15e ', Htp.c(1:3)));
fprintf('    PU center: [%s]\n', sprintf('%.15e ', PU.c(1:3)));
if isa(Pu, 'zonotope')
    fprintf('    Pu center: [%s]\n', sprintf('%.15e ', Pu.c(1:3)));
else
    fprintf('    Pu (numeric): [%s]\n', sprintf('%.15e ', Pu(1:3)));
end

% Manual addition
Htp_c = Htp.c;
if isa(Pu, 'zonotope')
    Pu_c = Pu.c;
else
    Pu_c = Pu;
end
PU_c = PU.c;
manual_sum_c = Htp_c + PU_c + Pu_c;
fprintf('  Manual sum of centers: [%s]\n', sprintf('%.15e ', manual_sum_c(1:3)));

Rtp = Htp + PU + Pu;
fprintf('  Rtp center (after addition): [%s]\n', sprintf('%.15e ', Rtp.c(1:3)));
fprintf('  Rtp generators shape: %dx%d\n', size(Rtp.G));
diff = Rtp.c - manual_sum_c;
fprintf('  Rtp center matches manual sum? %s\n', mat2str(all(abs(diff) < 1e-14)));
if any(abs(diff) >= 1e-14)
    fprintf('  Difference: [%s]\n', sprintf('%.15e ', diff(1:3)));
    fprintf('  Max abs diff: %.15e\n', max(abs(diff)));
end

% Step 8: Interval conversion
fprintf('\n[STEP 8] Interval Conversion\n');
fprintf('  Rtp center: [%s]\n', sprintf('%.15e ', Rtp.c(1:3)));
fprintf('  Rtp generators shape: %dx%d\n', size(Rtp.G));
if size(Rtp.G,2) > 0
    fprintf('  Rtp first generator: [%s]\n', sprintf('%.15e ', Rtp.G(:,1)));
end

delta = sum(abs(Rtp.G), 2);
fprintf('  delta: [%s]\n', sprintf('%.15e ', delta(1:3)));

IH_tp = interval(Rtp);
fprintf('  IH_tp.inf: [%s]\n', sprintf('%.15e ', IH_tp.inf(1:3)));
fprintf('  IH_tp.sup: [%s]\n', sprintf('%.15e ', IH_tp.sup(1:3)));

% Manual interval bounds
manual_inf = Rtp.c - delta;
manual_sup = Rtp.c + delta;
fprintf('  Manual inf: [%s]\n', sprintf('%.15e ', manual_inf(1:3)));
fprintf('  Manual sup: [%s]\n', sprintf('%.15e ', manual_sup(1:3)));
diff_inf = IH_tp.inf - manual_inf;
diff_sup = IH_tp.sup - manual_sup;
fprintf('  IH_tp.inf matches manual? %s\n', mat2str(all(abs(diff_inf) < 1e-14)));
fprintf('  IH_tp.sup matches manual? %s\n', mat2str(all(abs(diff_sup) < 1e-14)));

% Expected values (from previous runs)
fprintf('\n[STEP 9] Expected Final Values\n');
IH_tp_true_inf = [1.8057949711597598; 3.6433030183959114; 3.7940260617482671; 
                  1.9519553317477598; 9.3409949650858550; 4.0928655724716370];
IH_tp_true_sup = [2.2288356782079028; 4.0572873081850807; 4.1960714210115002; 
                  2.3451418924166987; 9.7630596270322201; 4.4862797486713282];

diff_inf = IH_tp.inf - IH_tp_true_inf;
diff_sup = IH_tp.sup - IH_tp_true_sup;
fprintf('  Difference inf: [%s]\n', sprintf('%.15e ', diff_inf));
fprintf('  Difference sup: [%s]\n', sprintf('%.15e ', diff_sup));
fprintf('  Max abs diff inf: %.15e\n', max(abs(diff_inf)));
fprintf('  Max abs diff sup: %.15e\n', max(abs(diff_sup)));

fprintf('\n================================================================================\n');
fprintf('MATLAB INVESTIGATION COMPLETE\n');
fprintf('================================================================================\n');
