function [state,entry] = finalizePredictionInjection( ...
    state,entry,POP,elite,extraRefinedElite,samplingMean,generation,BI, ...
    options,UFE,LFE)
%FINALIZEPREDICTIONINJECTION Archive actual evaluations and fit t+1 model.
if isempty(state)
    state = initializePredictionInjectionState(BI,options);
end

populationUF = [POP.UF];
populationFit = [POP.fit];
jointForUpdate = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
[~,fitnessOrder] = sort(-populationFit);
fitnessRank = NaN(numel(POP),1);
fitnessRank(fitnessOrder) = 1:numel(POP);
bestIndex = fitnessOrder(1);
previousBest = state.bestUF;
state.bestUF = max([state.bestUF,populationUF,elite.UF]);

entry.joint_for_cma_update = jointForUpdate;
entry.population_UF = populationUF(:);
entry.population_fit = populationFit(:);
entry.fitness_rank = fitnessRank;
entry.generation_best_index = bestIndex;
entry.generation_best_source = entry.source_labels{bestIndex};
entry.direct_is_generation_best = strcmp(entry.generation_best_source, ...
    'predicted_direct');
entry.auxiliary_is_generation_best = strcmp(entry.generation_best_source, ...
    'predicted_auxiliary');
entry.new_best_so_far = populationUF(bestIndex) > previousBest;
entry.best_so_far_UF = state.bestUF;
entry.UFE = UFE;
entry.LFE_total = LFE;
entry.elite_after_refinement_joint = [elite.UX,elite.LX];
entry.elite_after_refinement_UF = elite.UF;
entry.external_refinement_upper_evaluated = ~isempty(extraRefinedElite);
if ~isempty(extraRefinedElite)
    entry.external_refinement_joint = [extraRefinedElite.UX,extraRefinedElite.LX];
    entry.external_refinement_UF = extraRefinedElite.UF;
end
entry.new_best_so_far = state.bestUF > previousBest;
entry.external_refinement_is_new_best = ...
    entry.new_best_so_far && entry.external_refinement_upper_evaluated && ...
    extraRefinedElite.UF >= max(populationUF);

if options.usesPrediction
    rows = zeros(numel(POP),5);
    for i = 1:numel(POP)
        rows(i,:) = [generation,i,POP(i).UX,POP(i).UF];
    end
    if ~isempty(extraRefinedElite)
        rows(end+1,:) = [generation,numel(POP)+1, ...
            extraRefinedElite.UX,extraRefinedElite.UF];
    end
    state.history = [state.history;rows];
    state.history = state.history(state.history(:,1) >= ...
        generation-options.historyGenerations+1,:);
    model = fitLocalUpperQuadratic(state.history, ...
        samplingMean(1:BI.u_dim),generation,options);
    state.pending = struct('valid',model.valid,'x_pred',model.x_pred, ...
        'source_generation',generation,'model',model);
    entry.next_prediction_valid = model.valid;
    entry.next_prediction_reason = model.reason;
    entry.next_x_pred = model.x_pred;
    entry.next_prediction_model = model;
else
    entry.next_prediction_reason = 'disabled_A0';
end
end
