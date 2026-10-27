function entry = recordPredictionInjectionTrueEvaluation(entry,POP)
%RECORDPREDICTIONINJECTIONTRUEEVALUATION Read positions after normal LL/UL work.
entry.joint_after_true_evaluation = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
entry.upper_objective_after_true_evaluation = [POP.UF]';
end
