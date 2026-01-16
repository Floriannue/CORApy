% debug_matlab_plus_mtimes_pairs - capture MATLAB I/O pairs for plus/mtimes
% This script prints reference outputs for operations that were previously mismatched.

addpath('cora_matlab');

fprintf('MATLAB plus/mtimes pairs start\n');

% Interval + Zonotope (precedence check)
I = interval([-1; -1], [1; 1]);
c = [0; 0];
G = eye(2);
Z = zonotope([c, G]);

ZpI = Z + I;
IpZ = I + Z;

I_ZpI = interval(ZpI);
I_IpZ = interval(IpZ);

fprintf('Z+I interval inf = [%s]\n', sprintf('%.15g ', I_ZpI.inf));
fprintf('Z+I interval sup = [%s]\n', sprintf('%.15g ', I_ZpI.sup));
fprintf('I+Z interval inf = [%s]\n', sprintf('%.15g ', I_IpZ.inf));
fprintf('I+Z interval sup = [%s]\n', sprintf('%.15g ', I_IpZ.sup));

% Empty interval + Zonotope
try
    Iempty = interval.empty(2);
    ZpIempty = Z + Iempty;
    IpZempty = Iempty + Z;
    I_ZpIempty = interval(ZpIempty);
    I_IpZempty = interval(IpZempty);
    fprintf('Z+Iempty interval inf = [%s]\n', sprintf('%.15g ', I_ZpIempty.inf));
    fprintf('Z+Iempty interval sup = [%s]\n', sprintf('%.15g ', I_ZpIempty.sup));
    fprintf('Iempty+Z interval inf = [%s]\n', sprintf('%.15g ', I_IpZempty.inf));
    fprintf('Iempty+Z interval sup = [%s]\n', sprintf('%.15g ', I_IpZempty.sup));
catch ME
    fprintf('Empty interval + zonotope failed: %s\n', ME.message);
end

% Ellipsoid + Ellipsoid
Q1 = [5.4387811500952807, 12.4977183618314545; 12.4977183618314545, 29.6662117284481646];
q1 = [-0.7445068341257537; 3.5800647524843665];
Q2 = [4.2533342807136076, 0.6346400221575308; 0.6346400221575309, 0.0946946398147988];
q2 = [-2.4653656883489115; 0.2717868749873985];

E1 = ellipsoid(Q1, q1);
E2 = ellipsoid(Q2, q2);
Eout = E1 + E2;

fprintf('Ellipsoid+Ellipsoid Q = [%s]\n', sprintf('%.15g ', Eout.Q(:)'));
fprintf('Ellipsoid+Ellipsoid q = [%s]\n', sprintf('%.15g ', Eout.q(:)'));

fprintf('MATLAB plus/mtimes pairs end\n');
