% Debug script to test poly2bernstein edge cases in MATLAB

% Test 1: Empty generators
fprintf('=== Test 1: Empty generators ===\n');
try
    G = [];
    E = [];
    dom = interval([-1; -1], [1; 1]);
    [B, E_out] = poly2bernstein(G, E, dom);
    fprintf('Empty generators - B size: %s\n', mat2str(size(B)));
    fprintf('Empty generators - E_out size: %s\n', mat2str(size(E_out)));
catch ME
    fprintf('Empty generators - Error: %s\n', ME.message);
end

% Test 2: Zero polynomial
fprintf('\n=== Test 2: Zero polynomial ===\n');
try
    G = [0];
    E = [0; 0];
    dom = interval([-1; -1], [1; 1]);
    [B, E_out] = poly2bernstein(G, E, dom);
    fprintf('Zero polynomial - B: %s\n', mat2str(B));
    fprintf('Zero polynomial - E_out: %s\n', mat2str(E_out));
catch ME
    fprintf('Zero polynomial - Error: %s\n', ME.message);
end

% Test 3: Constant polynomial
fprintf('\n=== Test 3: Constant polynomial ===\n');
try
    G = [5];
    E = [0; 0];
    dom = interval([-1; -1], [1; 1]);
    [B, E_out] = poly2bernstein(G, E, dom);
    fprintf('Constant polynomial - B: %s\n', mat2str(B));
    fprintf('Constant polynomial - E_out: %s\n', mat2str(E_out));
catch ME
    fprintf('Constant polynomial - Error: %s\n', ME.message);
end

% Test 4: Simple polynomial
fprintf('\n=== Test 4: Simple polynomial ===\n');
try
    G = [1; 2];
    E = [1 0; 0 1];
    dom = interval([-1; -1], [1; 1]);
    [B, E_out] = poly2bernstein(G, E, dom);
    fprintf('Simple polynomial - B size: %s\n', mat2str(size(B)));
    fprintf('Simple polynomial - E_out size: %s\n', mat2str(size(E_out)));
catch ME
    fprintf('Simple polynomial - Error: %s\n', ME.message);
end
