function state = appendPredictionInjectionLog( ...
    state,entry,CMA,updateApplied,updateInfo,options)
%APPENDPREDICTIONINJECTIONLOG Store source-aware update and clipping evidence.
if nargin < 5 || isempty(updateInfo), updateInfo = struct(); end
entry.mean_update_applied = updateApplied;
entry.cma_mean_after_update = CMA.xmean;
entry.sigma_after_update = CMA.sigma;
entry.C_after_update = CMA.C;
if updateApplied
    entry.selected_parent_indices = updateInfo.eliteIndices;
    entry.selected_parent_joint = updateInfo.parentJoint;
    entry.selected_parent_sources = entry.source_labels(updateInfo.eliteIndices)';
    entry.direct_parent_selected = any(strcmp(entry.selected_parent_sources, ...
        'predicted_direct'));
    entry.auxiliary_parent_selected = any(strcmp(entry.selected_parent_sources, ...
        'predicted_auxiliary'));
    entry.parent_raw_steps = updateInfo.parentRawSteps;
    entry.parent_clipped_steps = updateInfo.parentClippedSteps;
    entry.parent_whitened_norms = updateInfo.parentWhitenedNorms;
    entry.parent_step_scales = updateInfo.parentStepScales;
    entry.injected_step_limited = ...
        ~strcmp(entry.selected_parent_sources,'normal') & ...
        updateInfo.parentStepScales' < 1;
    if options.consistencyCheckEnabled
        expected = entry.joint_for_cma_update(updateInfo.eliteIndices,:);
        sameShape = isequal(size(expected),size(updateInfo.parentJoint));
        if sameShape
            difference = max(abs(expected(:)-updateInfo.parentJoint(:)));
        else
            difference = Inf;
        end
        updatePasses = sameShape && all(isfinite(updateInfo.parentJoint(:))) && ...
            difference <= options.consistencyTolerance;
        previousPasses = ~entry.consistency_checked || ...
            logical(entry.consistency_passed);
        entry.consistency_checked = true;
        entry.consistency_passed = previousPasses && updatePasses;
        entry.consistency_message = sprintf('%s; update max difference %.3g', ...
            entry.consistency_message,difference);
        if ~entry.consistency_passed
            error('BLCMAES:PredictionInjectionInconsistency', ...
                'CMA-ES did not use the actual evaluated injected parents.');
        end
    end
end
if isempty(state.log)
    state.log = entry;
else
    state.log(end+1,1) = entry;
end
end
