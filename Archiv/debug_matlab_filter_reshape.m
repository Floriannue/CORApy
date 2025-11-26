% Test how MATLAB reshapes Filter for linFfilter

W = zeros(2,2,1,2);
W(:,:,1,1) = [1 -1; -1 2];
W(:,:,1,2) = [2 3; -1 -2];

fid = fopen('matlab_filter_reshape_output.txt', 'w');

fprintf(fid, 'Original Filter W:\n');
fprintf(fid, '  Shape: [%d %d %d %d]\n', size(W));
fprintf(fid, '  W(:,:,1,1):\n');
fprintf(fid, '    [%f %f]\n', W(:,:,1,1)');
fprintf(fid, '    [%f %f]\n', W(:,:,1,1)');
fprintf(fid, '  W(:,:,1,2):\n');
fprintf(fid, '    [%f %f]\n', W(:,:,1,2)');
fprintf(fid, '    [%f %f]\n', W(:,:,1,2)');
fprintf(fid, '\n');

% MATLAB: reshape(Filter,[],in_c,out_c)
in_c = 1;
out_c = 2;
W_reshaped = reshape(W, [], in_c, out_c);
fprintf(fid, 'After reshape(W,[],%d,%d):\n', in_c, out_c);
fprintf(fid, '  Shape: [%d %d %d]\n', size(W_reshaped));
fprintf(fid, '  W_reshaped(:,:,1) (first filter):\n');
for i = 1:size(W_reshaped, 1)
    fprintf(fid, '    [%f]\n', W_reshaped(i,:,1));
end
fprintf(fid, '  W_reshaped(:,:,2) (second filter):\n');
for i = 1:size(W_reshaped, 1)
    fprintf(fid, '    [%f]\n', W_reshaped(i,:,2));
end
fprintf(fid, '\n');

% Prepend zeros
linFfilter = [zeros(1,in_c,out_c,'like',W); W_reshaped];
fprintf(fid, 'After prepending zeros:\n');
fprintf(fid, '  Shape: [%d %d %d]\n', size(linFfilter));
fprintf(fid, '  linFfilter(:,:,1) (first filter):\n');
for i = 1:min(5, size(linFfilter, 1))
    fprintf(fid, '    [%f]\n', linFfilter(i,:,1));
end
fprintf(fid, '  linFfilter(:,:,2) (second filter):\n');
for i = 1:min(5, size(linFfilter, 1))
    fprintf(fid, '    [%f]\n', linFfilter(i,:,2));
end
fprintf(fid, '\n');

% Flatten
linFfilter_flat = linFfilter(:);
fprintf(fid, 'After flatten (linFfilter(:)):\n');
fprintf(fid, '  Length: %d\n', length(linFfilter_flat));
fprintf(fid, '  First 20 values: ');
fprintf(fid, '%f ', linFfilter_flat(1:min(20, length(linFfilter_flat))));
fprintf(fid, '\n');
fprintf(fid, '  All values: ');
fprintf(fid, '%f ', linFfilter_flat);
fprintf(fid, '\n');

fclose(fid);
fprintf('Results saved to matlab_filter_reshape_output.txt\n');

