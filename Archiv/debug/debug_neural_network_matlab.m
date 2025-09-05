% Debug script to trace neural network data flow in MATLAB
% This will help us understand how MATLAB handles the network

function debug_neural_network_matlab()
    fprintf('=== DEBUGGING NEURAL NETWORK DATA FLOW IN MATLAB ===\n');
    
    % Load the model
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx';
    fprintf('Loading model: %s\n', modelPath);
    
    % Read the ONNX network
    nn = neuralNetwork.readONNXNetwork(modelPath, true, 'BSSC');
    
    fprintf('\n=== NETWORK PROPERTIES ===\n');
    fprintf('neurons_in: %d\n', nn.neurons_in);
    fprintf('neurons_out: %d\n', nn.neurons_out);
    fprintf('Number of layers: %d\n', length(nn.layers));
    
    fprintf('\n=== LAYER DETAILS ===\n');
    for i = 1:length(nn.layers)
        layer = nn.layers{i};
        fprintf('Layer %d: %s\n', i-1, class(layer));
        
        % Check if layer has getNumNeurons method
        if ismethod(layer, 'getNumNeurons')
            try
                [nin, nout] = layer.getNumNeurons();
                fprintf('  Input neurons: %s, Output neurons: %s\n', mat2str(nin), mat2str(nout));
            catch ME
                fprintf('  getNumNeurons failed: %s\n', ME.message);
            end
        end
        
        % Check reshape layer specifically
        if isa(layer, 'nnReshapeLayer')
            fprintf('  Reshape idx_out: %s\n', mat2str(layer.idx_out));
            fprintf('  Reshape idx_out size: %s\n', mat2str(size(layer.idx_out)));
        end
        
        % Check linear layer specifically
        if isa(layer, 'nnLinearLayer')
            fprintf('  Linear layer W size: %s\n', mat2str(size(layer.W)));
            fprintf('  Linear layer b size: %s\n', mat2str(size(layer.b)));
            fprintf('  Linear layer expects %d inputs\n', size(layer.W, 2));
        end
    end
    
    fprintf('\n=== TESTING DATA FLOW ===\n');
    
    % Test with different input sizes
    test_sizes = {
        [5, 1],      % 5 inputs
        [50, 1],     % 50 inputs (matches neurons_in)
        [1, 1, 1, 5] % Original ONNX shape
    };
    
    for i = 1:length(test_sizes)
        test_size = test_sizes{i};
        fprintf('\n=== TESTING WITH INPUT SIZE %s ===\n', mat2str(test_size));
        
        try
            test_input = rand(test_size);
            fprintf('Input shape: %s\n', mat2str(size(test_input)));
            
            % Try to evaluate the network
            output = nn.evaluate(test_input);
            fprintf('SUCCESS: Output shape: %s\n', mat2str(size(output)));
            
        catch ME
            fprintf('FAILED: %s\n', ME.message);
        end
    end
    
    fprintf('\n=== TESTING LAYER BY LAYER ===\n');
    
    % Test each layer individually
    test_input = rand(5, 1);
    fprintf('Starting with input shape: %s\n', mat2str(size(test_input)));
    
    current_data = test_input;
    for i = 1:length(nn.layers)
        layer = nn.layers{i};
        fprintf('\n--- Layer %d: %s ---\n', i-1, class(layer));
        fprintf('Input shape: %s\n', mat2str(size(current_data)));
        
        try
            % Test the layer
            if ismethod(layer, 'evaluateNumeric')
                output = layer.evaluateNumeric(current_data, struct());
                fprintf('Output shape: %s\n', mat2str(size(output)));
                current_data = output;
            else
                fprintf('Layer has no evaluateNumeric method\n');
            end
        catch ME
            fprintf('ERROR in layer %d: %s\n', i-1, ME.message);
            fprintf('Layer type: %s\n', class(layer));
            if isprop(layer, 'W')
                fprintf('Layer W size: %s\n', mat2str(size(layer.W)));
            end
            if isprop(layer, 'b')
                fprintf('Layer b size: %s\n', mat2str(size(layer.b)));
            end
            break;
        end
    end
    
    fprintf('\n=== COMPARING WITH EXPECTATIONS ===\n');
    fprintf('MATLAB expects:\n');
    fprintf('- Input size: 5 (from neurons_in)\n');
    fprintf('- First layer should be reshape layer with idx_out: [1, 2, 3, 4, 5]\n');
    fprintf('- Second layer should be linear layer expecting 5 inputs\n');
    
    % Test the actual example
    fprintf('\n=== TESTING ACTUAL EXAMPLE ===\n');
    try
        % Load the example
        addpath('cora_python/examples/nn');
        example_neuralNetwork_verify_safe;
        fprintf('Example ran successfully!\n');
    catch ME
        fprintf('Example failed: %s\n', ME.message);
    end
end
