% Debug script to compare MATLAB network output with Python
% This will help identify where the discrepancy occurs

function debug_matlab_network()
    fprintf('=== DEBUGGING MATLAB NETWORK OUTPUT ===\n');
    
    % Load the same model
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx';
    fprintf('Loading MATLAB network from: %s\n', modelPath);
    
    % Read the ONNX network with same parameters as Python
    nn = neuralNetwork.readONNXNetwork(modelPath, true, 'BSSC');
    
    fprintf('\nMATLAB Network Properties:\n');
    fprintf('  neurons_in: %d\n', nn.neurons_in);
    fprintf('  neurons_out: %d\n', nn.neurons_out);
    fprintf('  Number of layers: %d\n', length(nn.layers));
    
    % Check first few linear layers
    linear_count = 0;
    for i = 1:length(nn.layers)
        layer = nn.layers{i};
        if isa(layer, 'nnLinearLayer')
            linear_count = linear_count + 1;
            fprintf('\nLayer %d (Linear %d):\n', i-1, linear_count);
            fprintf('  W shape: %s\n', mat2str(size(layer.W)));
            fprintf('  b shape: %s\n', mat2str(size(layer.b)));
            fprintf('  W first row: %s\n', mat2str(layer.W(1, 1:5), 8));
            fprintf('  b first values: %s\n', mat2str(layer.b(1:5)', 8));
            
            if linear_count >= 3
                break;
            end
        end
    end
    
    % Test with the counterexample input
    fprintf('\n=== TESTING WITH COUNTEREXAMPLE INPUT ===\n');
    matlab_x = [0.679858; 0.100000; 0.500000; 0.500000; -0.450000];
    fprintf('Input: %s\n', mat2str(matlab_x', 6));
    
    % Evaluate the network layer by layer
    fprintf('\n=== LAYER-BY-LAYER EVALUATION ===\n');
    layer_input = matlab_x;
    
    for i = 1:length(nn.layers)
        layer = nn.layers{i};
        layer_output = layer.evaluateNumeric(layer_input, struct());
        
        input_stats = sprintf('[%.3f, %.3f]', min(layer_input(:)), max(layer_input(:)));
        output_stats = sprintf('[%.3f, %.3f]', min(layer_output(:)), max(layer_output(:)));
        
        fprintf('Layer %d (%s): %s -> %s\n', i-1, class(layer), input_stats, output_stats);
        
        % Show actual values for key layers
        if i <= 5 || i >= length(nn.layers) - 1
            fprintf('  Input: %s\n', mat2str(layer_input(1:min(5, length(layer_input)))', 6));
            fprintf('  Output: %s\n', mat2str(layer_output(1:min(5, length(layer_output)))', 6));
        end
        
        layer_input = layer_output;
    end
    
    % Final network evaluation
    fprintf('\n=== FINAL NETWORK OUTPUT ===\n');
    final_output = nn.evaluate(matlab_x, struct());
    fprintf('MATLAB final output: %s\n', mat2str(final_output', 6));
    
    % Test specification
    fprintf('\n=== SPECIFICATION TEST ===\n');
    % Load the specification (from the Python example)
    A = [
        -1,  0,  0,  0,  0;
         0, -1,  0,  0,  0;
         0,  0, -1,  0,  0;
         0,  0,  0, -1,  0;
         0,  0,  0,  0, -1;
         1, -1,  0,  0,  0;
         1,  0, -1,  0,  0;
         1,  0,  0, -1,  0;
         1,  0,  0,  0, -1;
         0,  1, -1,  0,  0;
         0,  1,  0, -1,  0;
         0,  1,  0,  0, -1;
         0,  0,  1, -1,  0;
         0,  0,  1,  0, -1
    ];
    b = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    
    spec_result = A * final_output + b;
    fprintf('A * y + b: %s\n', mat2str(spec_result', 6));
    fprintf('All <= 0? %s\n', mat2str(all(spec_result <= 0)));
    fprintf('Max value: %.6f\n', max(spec_result));
    
    % This should be > 0 for counterexample (unsafe property)
    if any(spec_result > 0)
        fprintf('✓ MATLAB correctly finds counterexample (spec_result > 0)\n');
    else
        fprintf('✗ MATLAB does not find counterexample (all spec_result <= 0)\n');
    end
end
