function entry = recordPredictionInjectionEvaluationInput(entry,POP,options)
%RECORDPREDICTIONINJECTIONEVALUATIONINPUT Audit the lower-search input.
upperEnteringLowerSearch = cell2mat(arrayfun(@(p)p.UX,POP', ...
    'UniformOutput',false));
entry.upper_entering_lower_search = upperEnteringLowerSearch;
if ~options.consistencyCheckEnabled
    return;
end
tolerance = options.consistencyTolerance;
checks = [maxDifference(entry.joint_before_injection(:,1:size(entry.upper_before_injection,2)), ...
              entry.upper_before_injection), ...
          maxDifference(entry.joint_after_injection(:,1:size(entry.upper_after_injection,2)), ...
              entry.upper_after_injection), ...
          maxDifference(upperEnteringLowerSearch,entry.upper_after_injection)];
entry.consistency_checked = true;
entry.consistency_passed = all(checks <= tolerance);
entry.consistency_message = sprintf('sampling/injection/LL-input max difference %.3g', ...
    max(checks));
if ~entry.consistency_passed
    error('BLCMAES:PredictionInjectionInconsistency', ...
        'Injected candidate coordinates differ from the lower-search input.');
end
end

function difference = maxDifference(A,B)
if ~isequal(size(A),size(B)) || any(~isfinite(A(:))) || any(~isfinite(B(:)))
    difference = Inf;
else
    difference = max(abs(A(:)-B(:)));
    if isempty(difference), difference = 0; end
end
end
