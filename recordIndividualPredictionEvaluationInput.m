function entry = recordIndividualPredictionEvaluationInput(entry,POP,options)
%RECORDINDIVIDUALPREDICTIONEVALUATIONINPUT Capture the actual input to LL/UL.
% This is called immediately after corrected coordinates are copied to POP and
% immediately before lowerLevelSearch.  It only compares existing arrays.

jointEnteringEvaluation = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
entry.joint_entering_evaluation = jointEnteringEvaluation;
entry.upper_entering_lower_search = cell2mat(arrayfun(@(p)p.UX,POP', ...
    'UniformOutput',false));
if ~options.consistencyCheckEnabled
    return;
end

tolerance = options.consistencyTolerance;
checks = [maxAbsoluteDifference(entry.joint_before_correction(:, ...
              1:size(entry.upper_before,2)),entry.upper_before), ...
          maxAbsoluteDifference(entry.joint_after_correction(:, ...
              1:size(entry.upper_after,2)),entry.upper_after), ...
          maxAbsoluteDifference(jointEnteringEvaluation, ...
              entry.joint_after_correction), ...
          maxAbsoluteDifference(entry.upper_entering_lower_search, ...
              entry.upper_after), ...
          maxAbsoluteDifference(entry.upper_after-entry.upper_before, ...
              entry.correction_vectors)];
entry.consistency_checked = true;
entry.consistency_passed = all(checks <= tolerance);
entry.consistency_message = sprintf('sampling/evaluation max difference %.3g', ...
    max(checks));
if ~entry.consistency_passed
    error('BLCMAES:IndividualCorrectionInconsistency', ...
        'The correction log and the actual evaluation input differ: %s.', ...
        entry.consistency_message);
end
end

function difference = maxAbsoluteDifference(A,B)
if isempty(A) && isempty(B)
    difference = 0;
    return;
end
if ~isequal(size(A),size(B)) || any(~isfinite(A(:))) || any(~isfinite(B(:)))
    difference = Inf;
    return;
end
difference = max(abs(A(:)-B(:)));
if isempty(difference), difference = 0; end
end
