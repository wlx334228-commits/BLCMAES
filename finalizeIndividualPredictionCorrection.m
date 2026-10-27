function [state,entry] = finalizeIndividualPredictionCorrection( ...
    state,entry,POP,samplingMean,generation,BI,options,generationBestUF, ...
    eliteUF,UFE,LFE)
%FINALIZEINDIVIDUALPREDICTIONCORRECTION Archive and fit for generation t+1.
% The just-evaluated population is appended once.  The resulting model is
% stored as a pending prediction and cannot affect this same population.
if isempty(state)
    state = initializeIndividualPredictionState(BI,options);
end

populationUF = [POP.UF];
populationFit = [POP.fit];
state.bestUF = max([state.bestUF,populationUF,eliteUF]);
% upper_after remains the corrected point actually passed to lowerLevelSearch
% and UL_evaluate.  Refinement can subsequently replace one POP member, so
% retain that separate final population as the one consumed by updateCMAES.
jointForCMAUpdate = cell2mat(arrayfun(@(p)[p.UX p.LX],POP', ...
    'UniformOutput',false));
entry.upper_for_cma_update = jointForCMAUpdate(:,1:BI.u_dim);
entry.joint_for_cma_update = jointForCMAUpdate;
entry.generation_best_UF = generationBestUF;
entry.elite_UF = eliteUF;
entry.best_so_far_UF = state.bestUF;
entry.population_UF_min = min(populationUF);
entry.population_UF_median = median(populationUF);
entry.population_UF_std = std(populationUF,0);
entry.UFE = UFE;
entry.LFE_total = LFE;
[~,fitnessOrder] = sort(-populationFit);
fitnessRank = NaN(numel(POP),1);
fitnessRank(fitnessOrder) = 1:numel(POP);
entry.population_UF = populationUF(:);
entry.population_fit = populationFit(:);
entry.fitness_rank = fitnessRank;
transformed = entry.transformation_selected(:);
if any(transformed)
    entry.transformed_UF_median = median(populationUF(transformed));
    entry.transformed_rank_median = median(fitnessRank(transformed));
end
if any(~transformed)
    entry.untransformed_UF_median = median(populationUF(~transformed));
    entry.untransformed_rank_median = median(fitnessRank(~transformed));
end

if options.usesPrediction
    rows = zeros(numel(POP),5);
    for i = 1:numel(POP)
        rows(i,:) = [generation,i,POP(i).UX,POP(i).UF];
    end
    state.history = [state.history;rows];
    state.history = state.history(state.history(:,1) >= ...
        generation-options.historyGenerations+1,:);
    model = fitLocalUpperQuadratic(state.history,samplingMean,generation,options);
    state.pending = struct('valid',model.valid,'x_pred',model.x_pred, ...
        'source_generation',generation,'model',model);
    entry.next_prediction_valid = model.valid;
    entry.next_prediction_reason = model.reason;
    entry.next_x_pred = model.x_pred;
    entry.next_prediction_model = model;
else
    state.pending = struct('valid',false,'x_pred',NaN(1,BI.u_dim), ...
        'source_generation',NaN,'model',struct());
    if strcmp(options.mode,'A3')
        entry.next_prediction_reason = 'not_required_for_A3';
    else
        entry.next_prediction_reason = 'disabled';
    end
end
end
