function state = appendIndividualPredictionLog(state,entry,CMA,updateApplied,updateInfo,options)
%APPENDINDIVIDUALPREDICTIONLOG Save the complete per-generation state.
if nargin < 5 || isempty(updateInfo), updateInfo = struct(); end
if nargin < 6 || isempty(options)
    options = struct('consistencyCheckEnabled',false,'consistencyTolerance',1e-10);
end
entry.mean_update_applied = updateApplied;
if updateApplied
    % This is updateCMAES's result from the actual evaluated population.
    % No prediction code translated this mean after the update.
    entry.cma_mean_after_update = CMA.xmean(1:numel(entry.cma_mean_before_sampling));
    entry.selected_parent_indices = updateInfo.eliteIndices;
    entry.selected_parent_joint = updateInfo.parentJoint;
    entry.selected_parent_is_transformed = ...
        entry.transformation_selected(updateInfo.eliteIndices)';
    if options.consistencyCheckEnabled
        expectedParents = entry.joint_for_cma_update(updateInfo.eliteIndices,:);
        sameShape = isequal(size(expectedParents),size(updateInfo.parentJoint));
        if sameShape
            difference = max(abs(expectedParents(:)-updateInfo.parentJoint(:)));
        else
            difference = Inf;
        end
        updatePasses = sameShape && ...
            all(isfinite(updateInfo.parentJoint(:))) && ...
            difference <= options.consistencyTolerance;
        priorPasses = ~entry.consistency_checked || logical(entry.consistency_passed);
        entry.consistency_checked = true;
        entry.consistency_passed = priorPasses && updatePasses;
        entry.consistency_message = sprintf('%s; update max difference %.3g', ...
            entry.consistency_message,difference);
        if ~entry.consistency_passed
            error('BLCMAES:IndividualCorrectionInconsistency', ...
                'CMA-ES did not use the logged evaluated parent positions.');
        end
    end
end
if isempty(state.log)
    state.log = entry;
else
    state.log(end+1,1) = entry;
end
end
