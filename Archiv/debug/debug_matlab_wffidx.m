% Debug script to get MATLAB WffIdx values

W = zeros(2, 2, 1, 2);
W(:, :, 1, 1) = [1, -1; -1, 2];
W(:, :, 1, 2) = [2, 3; -1, -2];

layer = nnConv2DLayer(W, [1.0; -2.0]);
nn = neuralNetwork({layer});
nn.setInputSize([4, 4, 1]);

% Get WffIdx
[WffIdx, ~] = layer.aux_computeWeightMatIdx();

fid = fopen('matlab_wffidx_output.txt', 'w');

fprintf(fid, 'MATLAB WffIdx (first 5x5):\n');
for i = 1:5
    fprintf(fid, '  ');
    fprintf(fid, '%.1f ', WffIdx(i, 1:5));
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% Get linFfilter
Wff = layer.aux_conv2Mat();
lin_layer = layer.convert2nnLinearLayer();
% We can't directly get linFfilter, but we can verify Wff

fprintf(fid, 'MATLAB Wff (first row, first 5):\n');
fprintf(fid, '  ');
fprintf(fid, '%.6f ', Wff(1, 1:5));
fprintf(fid, '\n');

fclose(fid);
fprintf('Results saved to matlab_wffidx_output.txt\n');

