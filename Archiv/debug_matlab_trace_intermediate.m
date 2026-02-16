% MATLAB function to add intermediate value tracking to linReach_adaptive
% This modifies the inner loop to log all key intermediate values
% 
% Usage: Add this code to linReach_adaptive.m or create a wrapper function

function [Rti, Rtp, options] = linReach_adaptive_traced(nlnsys, Rstart, params, options)
% Wrapper that adds tracing to linReach_adaptive

% Enable tracing
options.traceIntermediateValues = true;

% Call original function
[Rti, Rtp, options] = linReach_adaptive(nlnsys, Rstart, params, options);

end

% To add tracing directly to linReach_adaptive.m, add this code in the inner loop:
% 
% % Track intermediate values
% if isfield(options, 'traceIntermediateValues') && options.traceIntermediateValues
%     fid = fopen(sprintf('intermediate_values_step%d_inner_loop.txt', options.i), 'a');
%     if fid > 0
%         fprintf(fid, '\n--- Inner Loop Iteration %d ---\n', inner_iter);
%         fprintf(fid, 'error_adm: [%.15e', error_adm(1));
%         for i=2:length(error_adm), fprintf(fid, ' %.15e', error_adm(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'error_adm_max: %.15e\n', max(error_adm));
%         
%         % Track RallError
%         RallError_center = center(RallError);
%         RallError_gens = generators(RallError);
%         RallError_radius = sum(abs(RallError_gens), 2);
%         RallError_radius_max = max(RallError_radius);
%         fprintf(fid, 'RallError center: [%.15e', RallError_center(1));
%         for i=2:length(RallError_center), fprintf(fid, ' %.15e', RallError_center(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'RallError radius: [%.15e', RallError_radius(1));
%         for i=2:length(RallError_radius), fprintf(fid, ' %.15e', RallError_radius(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'RallError radius_max: %.15e\n', RallError_radius_max);
%         
%         % Track Rmax
%         Rmax_center = center(Rmax);
%         Rmax_gens = generators(Rmax);
%         Rmax_radius = sum(abs(Rmax_gens), 2);
%         Rmax_radius_max = max(Rmax_radius);
%         fprintf(fid, 'Rmax center: [%.15e', Rmax_center(1));
%         for i=2:length(Rmax_center), fprintf(fid, ' %.15e', Rmax_center(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'Rmax radius: [%.15e', Rmax_radius(1));
%         for i=2:length(Rmax_radius), fprintf(fid, ' %.15e', Rmax_radius(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'Rmax radius_max: %.15e\n', Rmax_radius_max);
%         
%         % Track trueError and perfIndCurr (after priv_abstractionError_adaptive)
%         fprintf(fid, 'trueError: [%.15e', trueError(1));
%         for i=2:length(trueError), fprintf(fid, ' %.15e', trueError(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'trueError_max: %.15e\n', max(trueError));
%         
%         perfIndCurr_ratio = trueError ./ error_adm;
%         perfIndCurr = max(perfIndCurr_ratio);
%         fprintf(fid, 'perfIndCurr_ratio: [%.15e', perfIndCurr_ratio(1));
%         for i=2:length(perfIndCurr_ratio), fprintf(fid, ' %.15e', perfIndCurr_ratio(i)); end
%         fprintf(fid, ']\n');
%         fprintf(fid, 'perfIndCurr: %.15e\n', perfIndCurr);
%         fprintf(fid, 'perfIndCurr isinf: %d\n', isinf(perfIndCurr));
%         fprintf(fid, 'perfIndCurr isnan: %d\n', isnan(perfIndCurr));
%         fprintf(fid, 'perfIndCurr <= 1: %d\n', perfIndCurr <= 1);
%         
%         fprintf(fid, 'perfIndCounter: %d\n', perfIndCounter);
%         fprintf(fid, 'perfInds: [');
%         if ~isempty(perfInds)
%             fprintf(fid, '%.15e', perfInds(1));
%             for i=2:length(perfInds), fprintf(fid, ' %.15e', perfInds(i)); end
%         end
%         fprintf(fid, ']\n');
%         
%         fclose(fid);
%     end
% end
