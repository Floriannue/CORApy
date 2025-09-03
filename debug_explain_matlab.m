% Debug script to test MATLAB explain method
clear; clc;

% construct network
W1 = [ -9 -8 -7 ; 10 -6 0 ; -6 2 5 ; 4 4 -8 ; -5 -8 2 ; 0 6 2 ; -7 10 -2 ; 0 8 6 ; 1 -3 -2 ; 3 9 2 ];
W2 = [ 3 6 -5 3 -6 2 6 2 -4 8 ; 4 1 7 -3 -4 4 2 0 2 -1 ; -3 9 1 5 10 9 1 4 -6 -7 ];
nn = neuralNetwork({nnLinearLayer(W1),nnSigmoidLayer(),nnLinearLayer(W2)});

% construct input
x = [1;2;3];
label = 1;

% compute explanation
verbose = false;
epsilon = 0.2;

% method: standard
method = 'standard';
fprintf('MATLAB Results for method: %s\n', method);
[idxFreedFeatsStandard, featOrder, timesPerFeat] = nn.explain(x,label,epsilon,'InputSize',[3,1,1],'Method',method,'Verbose',verbose);

fprintf('idxFreedFeatsStandard: %s\n', mat2str(idxFreedFeatsStandard));
fprintf('featOrder: %s\n', mat2str(featOrder));
fprintf('timesPerFeat: %s\n', mat2str(timesPerFeat));

% method: abstract+refine
method = 'abstract+refine';
fprintf('\nMATLAB Results for method: %s\n', method);
[idxFreedFeatsAbstract, featOrder2, timesPerFeat2] = nn.explain(x,label,epsilon,'InputSize',[3,1,1],'Method',method,'Verbose',verbose);

fprintf('idxFreedFeatsAbstract: %s\n', mat2str(idxFreedFeatsAbstract));
fprintf('featOrder2: %s\n', mat2str(featOrder2));
fprintf('timesPerFeat2: %s\n', mat2str(timesPerFeat2));

% Test with simple network
fprintf('\n=== Simple Network Test ===\n');
W1_simple = [1 2; 3 4];
W2_simple = [1 0; 0 1];
nn_simple = neuralNetwork({nnLinearLayer(W1_simple),nnSigmoidLayer(),nnLinearLayer(W2_simple)});

x_simple = [1;2];
label_simple = 1;  % Changed from 0 to 1 to avoid indexing error
epsilon_simple = 0.2;

try
    [idxFreedFeatsSimple, featOrderSimple, timesPerFeatSimple] = nn_simple.explain(x_simple,label_simple,epsilon_simple,'InputSize',[2,1,1],'Method','standard','Verbose',false);
    
    fprintf('Simple network idxFreedFeats: %s\n', mat2str(idxFreedFeatsSimple));
    fprintf('Simple network featOrder: %s\n', mat2str(featOrderSimple));
    fprintf('Simple network timesPerFeat: %s\n', mat2str(timesPerFeatSimple));
catch ME
    fprintf('Simple network test failed: %s\n', ME.message);
end
