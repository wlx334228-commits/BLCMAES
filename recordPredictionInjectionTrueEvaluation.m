function entry = recordPredictionInjectionTrueEvaluation(entry,POP)
%RECORDPREDICTIONINJECTIONTRUEEVALUATION Read positions after normal LL/UL work.
entry.joint_after_true_evaluation = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
entry.upper_objective_after_true_evaluation = [POP.UF]';
% These are the exact upper-evaluation counters immediately after each
% sampled member's one real bilevel evaluation.  They let an offline branch
% study reconstruct the historical best at an identical UFE checkpoint even
% when optional refinement produces different extra evaluations by branch.
entry.candidate_upper_evaluation_UFE = [POP.UFEs]';
end
