% Debug script to trace MATLAB verify behavior for prop_2
% This will help identify where Python translation differs

CORAROOT = getenv('CORAROOT');
if isempty(CORAROOT)
    error('CORAROOT environment variable not set');
end

modelPath = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
prop2Filename = [CORAROOT '/models/Cora/nn/prop_2.vnnlib'];

% Read network and options (simplified version)
nn = neuralNetwork.readONNXNetwork(modelPath);
[nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,prop2Filename);

fprintf('=== MATLAB VERIFY DEBUG FOR PROP_2 ===\n');
fprintf('safeSet: %d\n', safeSet);
fprintf('x shape: [%d, %d]\n', size(x,1), size(x,2));
fprintf('r shape: [%d, %d]\n', size(r,1), size(r,2));
fprintf('A shape: [%d, %d]\n', size(A,1), size(A,2));
fprintf('b shape: [%d, %d]\n', size(b,1), size(b,2));

timeout = 100;
nSplits = 5;
nDims = 1;
totalNumSplits = 0;
verifiedPatches = 0;

bs = options.nn.train.mini_batch_size;
inputDataClass = single(1);
useGpu = options.nn.train.use_gpu;
if useGpu
    inputDataClass = gpuArray(inputDataClass);
end
nn.castWeights(inputDataClass);

idxLayer = 1:length(nn.layers);
numGen = nn.prepareForZonoBatchEval(x,options,idxLayer);
idMat = cast([eye(size(x,1)) zeros(size(x,1),numGen - size(x,1))], 'like',inputDataClass);
batchG = cast(repmat(idMat,1,1,bs),'like',inputDataClass);

xs = x;
rs = r;
n0 = size(x,1);
res = [];

timerVal = tic;
iteration = 0;

while size(xs,2) > 0
    iteration = iteration + 1;
    time = toc(timerVal);
    if time > timeout
        res = 'UNKNOWN';
        break;
    end
    
    fprintf('\n=== ITERATION %d ===\n', iteration);
    fprintf('Queue size: %d\n', size(xs,2));
    
    % Pop next batch
    [xi,ri,xs,rs] = aux_pop(xs,rs,bs);
    xi = cast(xi,'like',inputDataClass);
    ri = cast(ri,'like',inputDataClass);
    
    fprintf('Popped batch size: %d\n', size(xi,2));
    fprintf('xi(:,1):\n');
    disp(xi(:,1));
    fprintf('ri(:,1):\n');
    disp(ri(:,1));
    
    % Falsification
    [S,~] = nn.calcSensitivity(xi,options,false);
    S = max(S,1e-3);
    
    fprintf('S shape: [%d, %d, %d]\n', size(S,1), size(S,2), size(S,3));
    fprintf('S(:,1,1) (first output, all inputs, first batch):\n');
    disp(S(:,1,1));
    
    % MATLAB: sens = permute(sum(abs(S)),[2 1 3]); sens = sens(:,:);
    sens_temp = sum(abs(S));  % Sum over first dimension (outputs)
    fprintf('sum(abs(S)) shape: [%d, %d]\n', size(sens_temp,1), size(sens_temp,2));
    sens = permute(sens_temp,[2 1 3]);
    fprintf('permute(sum(abs(S)),[2 1 3]) shape: [%d, %d, %d]\n', size(sens,1), size(sens,2), size(sens,3));
    sens = sens(:,:);
    fprintf('sens(:,:) final shape: [%d, %d]\n', size(sens,1), size(sens,2));
    fprintf('sens(:,1) (first batch):\n');
    disp(sens(:,1));
    
    zi = xi + ri.*sign(sens);
    fprintf('zi(:,1) (adversarial input):\n');
    disp(zi(:,1));
    
    yi = nn.evaluate_(zi,options,idxLayer);
    fprintf('yi shape: [%d, %d]\n', size(yi,1), size(yi,2));
    fprintf('yi(:,1) (output for adversarial input):\n');
    disp(yi(:,1));
    
    if safeSet
        checkSpecs = any(A*yi + b >= 0,1);
    else
        checkSpecs = all(A*yi + b <= 0,1);
    end
    
    fprintf('A*yi + b shape: [%d, %d]\n', size(A*yi + b,1), size(A*yi + b,2));
    fprintf('A*yi(:,1) + b (first batch):\n');
    disp(A*yi(:,1) + b);
    fprintf('checkSpecs:\n');
    disp(checkSpecs);
    fprintf('any(checkSpecs): %d\n', any(checkSpecs));
    
    if any(checkSpecs)
        fprintf('FOUND COUNTEREXAMPLE!\n');
        idNzEntry = find(checkSpecs);
        id = idNzEntry(1);
        x_ = zi(:,id);
        nn.castWeights(single(1));
        y_ = nn.evaluate_(gather(x_),options,idxLayer);
        fprintf('Counterexample x_:\n');
        disp(x_);
        fprintf('Counterexample y_:\n');
        disp(y_);
        res = 'COUNTEREXAMPLE';
        break;
    end
    
    % Limit iterations for debugging
    if iteration >= 3
        fprintf('Stopping after 3 iterations for debugging\n');
        break;
    end
end

fprintf('\n=== FINAL RESULT ===\n');
fprintf('res: %s\n', res);

