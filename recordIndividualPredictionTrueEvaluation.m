function entry = recordIndividualPredictionTrueEvaluation(entry,POP,options)
%RECORDINDIVIDUALPREDICTIONTRUEEVALUATION Log evaluated LL/UL coordinates.
% Called after every normal lowerLevelSearch and UL_evaluate call but before
% fitness assignment/refinement.  It observes fields already computed by the
% algorithm and makes no objective or random-number call.

jointAfterTrueEvaluation = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
entry.joint_after_true_evaluation = jointAfterTrueEvaluation;
entry.upper_objective_after_true_evaluation = [POP.UF]';
if ~options.consistencyCheckEnabled
    return;
end

upperForLowerSearch = jointAfterTrueEvaluation(:,1:size(entry.upper_after,2));
entry.upper_entering_lower_search = upperForLowerSearch;
if ~isequal(size(upperForLowerSearch),size(entry.upper_after)) || ...
        any(~isfinite(upperForLowerSearch(:))) || ...
        max(abs(upperForLowerSearch(:)-entry.upper_after(:))) > ...
            options.consistencyTolerance
    entry.consistency_passed = false;
    entry.consistency_message = sprintf('%s; lower-search upper input mismatch', ...
        entry.consistency_message);
    error('BLCMAES:IndividualCorrectionInconsistency', ...
        'The lower-level search did not receive the corrected upper point.');
end
entry.consistency_message = sprintf('%s; LL/UL upper input consistent', ...
    entry.consistency_message);
end
