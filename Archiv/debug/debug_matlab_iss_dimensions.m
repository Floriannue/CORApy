% Debug script to check dimensions in MATLAB for ISS test case
% Compare with Python implementation

% Load system matrices
load iss.mat A B C

% Display original dimensions
fprintf('Original system:\n');
fprintf('  A shape: %d x %d\n', size(A,1), size(A,2));
fprintf('  B shape: %d x %d\n', size(B,1), size(B,2));
fprintf('  C shape: %d x %d\n', size(C,1), size(C,2));

% Construct extended system matrices (inputs as additional states)
dim_x = length(A);
A_  = [A,B;zeros(size(B,2),dim_x + size(B,2))];
B_  = zeros(dim_x+size(B,2),1);
C_  = [C,zeros(size(C,1),size(B,2))];

fprintf('\nExtended system:\n');
fprintf('  A_ shape: %d x %d\n', size(A_,1), size(A_,2));
fprintf('  B_ shape: %d x %d\n', size(B_,1), size(B_,2));
fprintf('  C_ shape: %d x %d\n', size(C_,1), size(C_,2));

% Construct the linear system object
sys = linearSys('iss',A_,B_,[],C_);

fprintf('\nLinearSys object:\n');
fprintf('  sys.nrOfDims: %d\n', sys.nrOfDims);
fprintf('  sys.nrOfInputs: %d\n', sys.nrOfInputs);
fprintf('  sys.nrOfOutputs: %d\n', sys.nrOfOutputs);

% Create specifications
d = 5e-4;
P1 = polytope([0 0 1],-d);
P2 = polytope([0 0 -1],-d);
spec = specification({P1,P2},'unsafeSet');

fprintf('\nSpecifications:\n');
fprintf('  P1.A shape: %d x %d\n', size(P1.A,1), size(P1.A,2));
fprintf('  P2.A shape: %d x %d\n', size(P2.A,1), size(P2.A,2));

% Check specifications directly
fprintf('\nSpecification details:\n');
fprintf('  spec(1).set.A shape: %d x %d\n', size(spec(1).set.A,1), size(spec(1).set.A,2));
fprintf('  spec(2).set.A shape: %d x %d\n', size(spec(2).set.A,1), size(spec(2).set.A,2));

% Simulate what happens in priv_verifyRA_supportFunc
% For unsafe sets, we use -spec.set.A
fprintf('\nSimulating Cs construction:\n');
nrSpecs = 2; % Two unsafe sets
Cs = zeros(nrSpecs, sys.nrOfOutputs);
fprintf('  Cs shape (before assignment): %d x %d\n', size(Cs,1), size(Cs,2));
fprintf('  sys.nrOfOutputs: %d\n', sys.nrOfOutputs);

% Assign like MATLAB does
Cs(1,:) = -spec(1).set.A;
Cs(2,:) = -spec(2).set.A;

fprintf('  Cs shape (after assignment): %d x %d\n', size(Cs,1), size(Cs,2));
fprintf('  Cs contents:\n');
disp(Cs);

% Check params.V initialization
params.U = zonotope(0);
params.R0 = zonotope([interval(-0.0001*ones(270,1),0.0001*ones(270,1)); ...
      interval([0;0.8;0.9],[0.1;1;1])]);
params.tFinal = 20;

fprintf('\nBefore aux_canonicalForm:\n');
fprintf('  params has V: %d\n', isfield(params, 'V'));

% Call aux_canonicalForm (simulate what happens)
% Note: This is a simplified version
if ~isfield(params, 'W')
    params.W = zonotope(zeros(sys.nrOfDims,1));
end
if ~isfield(params, 'V')
    params.V = zonotope(zeros(sys.nrOfOutputs,1));
end

fprintf('  params.V dimension: %d\n', params.V.dim());
fprintf('  sys.nrOfOutputs: %d\n', sys.nrOfOutputs);
fprintf('  Cs columns: %d\n', size(Cs,2));

fprintf('\nKey observation:\n');
fprintf('  In MATLAB, Cs is created with sys.nrOfOutputs = %d columns\n', sys.nrOfOutputs);
fprintf('  Specifications have %d columns\n', size(spec(1).set.A,2));
fprintf('  They should match for assignment to work correctly.\n');
fprintf('  If dimensions match, MATLAB assignment works directly.\n');
