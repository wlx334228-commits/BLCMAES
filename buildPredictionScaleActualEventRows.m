function [eventRow,candidateRows] = buildPredictionScaleActualEventRows( ...
    state,smdNo,runNo,runSeed,multiScaleDiagnosed)
%BUILDPREDICTIONSCALEACTUALEVENTROWS Convert one real auxiliary event to CSV rows.
% All values originate from the completed main run.  This helper does not
% sample, evaluate, rank, or update the optimizer.

upperDim = numel(state.x_pred);
auxCount = numel(state.auxiliary_indices);
assert(upperDim == 2 && auxCount > 0, ...
    'This first all-event analysis expects at least one 2-D auxiliary candidate.');
Cuu = (state.C(1:upperDim,1:upperDim)+state.C(1:upperDim,1:upperDim)')/2;
cmaRmsScale = state.sigma*sqrt(trace(Cuu)/upperDim);
auxStep = state.eta*state.sigma;
auxRmsScale = state.eta*cmaRmsScale;
eventId = stateIdentifier(smdNo,runNo,state);
normalUF = state.normal_UF(:)';
candidateRows = repmat(emptyCandidateRow(),auxCount,1);
for index = 1:auxCount
    populationIndex = state.auxiliary_indices(index);
    trueJoint = state.auxiliary_joint(index,:);
    sampledJoint = state.auxiliary_sampled_joint(index,:);
    rawJoint = state.auxiliary_raw_joint(index,:);
    updateJoint = state.population_joint(populationIndex,:);
    actualOffset = sampledJoint(1:upperDim)-state.x_pred;
    rawOffset = rawJoint(1:upperDim)-state.x_pred;
    parentSelected = NaN;
    if state.mean_update_applied
        parentSelected = double(any(state.selected_parent_indices == populationIndex));
    end
    row = emptyCandidateRow();
    row.SMD = smdNo; row.run_no = runNo; row.seed = runSeed;
    row.event_id = eventId; row.event_ordinal = state.event_ordinal;
    row.stage = state.stage; row.generation = state.generation;
    row.log_index = state.log_index; row.progress = state.progress;
    row.UFE_start = state.UFE_start; row.UFE_end = state.UFE_end;
    row.main_UFE_before_sampling = state.UFE_before_sampling;
    row.main_LFE_before_sampling = state.LFE_before_sampling;
    row.main_UFE_after_population = state.UFE_after_population;
    row.main_LFE_after_population = state.LFE_after_population;
    row.UFE_event = state.UFE_after_population;
    row.LFE_event = state.LFE_after_population;
    row.x_pred_1 = state.x_pred(1); row.x_pred_2 = state.x_pred(2);
    row.cma_mean_1 = state.cma_mean(1); row.cma_mean_2 = state.cma_mean(2);
    row.sigma = state.sigma; row.eta = state.eta;
    row.auxiliary_step = auxStep; row.cma_rms_scale = cmaRmsScale;
    row.auxiliary_rms_scale = auxRmsScale;
    row.Cuu_11 = Cuu(1,1); row.Cuu_12 = Cuu(1,2); row.Cuu_22 = Cuu(2,2);
    row.auxiliary_index = index; row.population_index = populationIndex;
    row.raw_xu_1 = rawJoint(1); row.raw_xu_2 = rawJoint(2);
    row.actual_xu_1 = sampledJoint(1); row.actual_xu_2 = sampledJoint(2);
    row.true_xu_1 = trueJoint(1); row.true_xu_2 = trueJoint(2);
    row.true_xl_1 = trueJoint(3); row.true_xl_2 = trueJoint(4); row.true_xl_3 = trueJoint(5);
    % Refinement is part of the original algorithm and can replace one
    % population member before CMA-ES selection.  Preserve that exact
    % update-side position separately from the first true evaluation above.
    row.cma_update_xu_1 = updateJoint(1); row.cma_update_xu_2 = updateJoint(2);
    row.cma_update_xl_1 = updateJoint(3); row.cma_update_xl_2 = updateJoint(4);
    row.cma_update_xl_3 = updateJoint(5);
    row.cma_update_UF = state.population_UF(populationIndex);
    row.cma_update_position_changed = double(any(abs(updateJoint-trueJoint) > 0));
    row.boundary_repair_count = sum(state.auxiliary_boundary_repaired(index,:));
    row.raw_distance_to_prediction = norm(rawOffset);
    row.actual_distance_to_prediction = norm(actualOffset);
    row.relative_cma_distance = safeDivide(norm(actualOffset),cmaRmsScale);
    row.standardized_distance = standardisedDistance(actualOffset,state.eta,state.sigma,Cuu);
    row.auxiliary_UF = state.auxiliary_UF(index);
    row.direct_true_UF = state.direct_UF;
    row.delta_UF_to_direct = row.auxiliary_UF-state.direct_UF;
    row.better_than_direct = double(row.auxiliary_UF > state.direct_UF);
    row.normal_candidate_count = numel(normalUF);
    row.normal_UF_max = max(normalUF); row.normal_UF_median = median(normalUF);
    row.better_than_normal_pair_count = sum(row.auxiliary_UF > normalUF);
    row.better_than_normal_pair_rate = row.better_than_normal_pair_count/numel(normalUF);
    row.better_than_all_normal = double(row.better_than_normal_pair_count == numel(normalUF));
    row.parent_selection_applicable = double(state.mean_update_applied);
    row.parent_selected = parentSelected;
    row.history_best_UF_before_sampling = state.history_best_UF_before_sampling;
    row.better_than_history_best = double(row.auxiliary_UF > ...
        state.history_best_UF_before_sampling);
    row.multi_scale_diagnosed = double(multiScaleDiagnosed);
    candidateRows(index) = row;
end

eventRow = emptyEventRow();
eventRow.SMD = smdNo; eventRow.run_no = runNo; eventRow.seed = runSeed;
eventRow.event_id = eventId; eventRow.event_ordinal = state.event_ordinal;
eventRow.stage = state.stage; eventRow.generation = state.generation;
eventRow.log_index = state.log_index; eventRow.progress = state.progress;
eventRow.UFE_start = state.UFE_start; eventRow.UFE_end = state.UFE_end;
eventRow.main_UFE_before_sampling = state.UFE_before_sampling;
eventRow.main_LFE_before_sampling = state.LFE_before_sampling;
eventRow.main_UFE_after_population = state.UFE_after_population;
eventRow.main_LFE_after_population = state.LFE_after_population;
eventRow.UFE_event = state.UFE_after_population;
eventRow.LFE_event = state.LFE_after_population;
eventRow.x_pred_1 = state.x_pred(1); eventRow.x_pred_2 = state.x_pred(2);
eventRow.direct_true_UF = state.direct_UF;
eventRow.cma_mean_1 = state.cma_mean(1); eventRow.cma_mean_2 = state.cma_mean(2);
eventRow.sigma = state.sigma; eventRow.eta = state.eta;
eventRow.auxiliary_step = auxStep; eventRow.cma_rms_scale = cmaRmsScale;
eventRow.auxiliary_rms_scale = auxRmsScale;
eventRow.Cuu_11 = Cuu(1,1); eventRow.Cuu_12 = Cuu(1,2); eventRow.Cuu_22 = Cuu(2,2);
eventRow.auxiliary_candidate_count = auxCount;
eventRow.auxiliary_UF_best = max([candidateRows.auxiliary_UF]);
eventRow.auxiliary_UF_median = median([candidateRows.auxiliary_UF]);
eventRow.auxiliary_better_than_direct_rate = mean([candidateRows.better_than_direct]);
eventRow.auxiliary_better_than_normal_pair_rate = ...
    mean([candidateRows.better_than_normal_pair_rate]);
eventRow.auxiliary_parent_rate = meanIgnoringNaN([candidateRows.parent_selected]);
eventRow.auxiliary_history_improvement_rate = ...
    mean([candidateRows.better_than_history_best]);
eventRow.boundary_repairs = sum([candidateRows.boundary_repair_count]);
eventRow.mean_update_applied = double(state.mean_update_applied);
eventRow.multi_scale_diagnosed = double(multiScaleDiagnosed);
for row = 1:size(state.C,1)
    for column = 1:size(state.C,2)
        eventRow.(sprintf('C_%d_%d',row,column)) = state.C(row,column);
    end
end
end

function value = safeDivide(numerator,denominator)
if denominator > 0 && isfinite(denominator), value = numerator/denominator;
else, value = NaN; end
end

function value = standardisedDistance(offset,eta,sigma,covariance)
if ~(eta > 0 && sigma > 0), value = NaN; return; end
[vectors,values] = eig((covariance+covariance')/2);
eigenvalues = real(diag(values));
if any(~isfinite(eigenvalues)) || any(eigenvalues <= 0), value = NaN; return; end
factor = vectors*diag(sqrt(eigenvalues));
value = norm((offset/(eta*sigma))/factor');
end

function value = meanIgnoringNaN(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = mean(values); end
end

function id = stateIdentifier(smdNo,runNo,state)
id = sprintf('SMD%d_run%02d_gen%04d',smdNo,runNo,state.generation);
end

function row = emptyEventRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'event_id','', ...
    'event_ordinal',NaN,'stage','','generation',NaN,'log_index',NaN, ...
    'progress',NaN,'UFE_start',NaN,'UFE_end',NaN, ...
    'main_UFE_before_sampling',NaN,'main_LFE_before_sampling',NaN, ...
    'main_UFE_after_population',NaN,'main_LFE_after_population',NaN, ...
    'UFE_event',NaN,'LFE_event',NaN, ...
    'x_pred_1',NaN,'x_pred_2',NaN,'direct_true_UF',NaN, ...
    'cma_mean_1',NaN,'cma_mean_2',NaN,'sigma',NaN,'eta',NaN, ...
    'auxiliary_step',NaN,'cma_rms_scale',NaN,'auxiliary_rms_scale',NaN, ...
    'Cuu_11',NaN,'Cuu_12',NaN,'Cuu_22',NaN, ...
    'auxiliary_candidate_count',NaN,'auxiliary_UF_best',NaN, ...
    'auxiliary_UF_median',NaN,'auxiliary_better_than_direct_rate',NaN, ...
    'auxiliary_better_than_normal_pair_rate',NaN, ...
    'auxiliary_parent_rate',NaN,'auxiliary_history_improvement_rate',NaN, ...
    'boundary_repairs',NaN,'mean_update_applied',NaN,'multi_scale_diagnosed',NaN);
for index = 1:5
    for column = 1:5
        row.(sprintf('C_%d_%d',index,column)) = NaN;
    end
end
end

function row = emptyCandidateRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'event_id','', ...
    'event_ordinal',NaN,'stage','','generation',NaN,'log_index',NaN, ...
    'progress',NaN,'UFE_start',NaN,'UFE_end',NaN, ...
    'main_UFE_before_sampling',NaN,'main_LFE_before_sampling',NaN, ...
    'main_UFE_after_population',NaN,'main_LFE_after_population',NaN, ...
    'UFE_event',NaN,'LFE_event',NaN, ...
    'x_pred_1',NaN,'x_pred_2',NaN,'cma_mean_1',NaN,'cma_mean_2',NaN, ...
    'sigma',NaN,'eta',NaN,'auxiliary_step',NaN,'cma_rms_scale',NaN, ...
    'auxiliary_rms_scale',NaN,'Cuu_11',NaN,'Cuu_12',NaN,'Cuu_22',NaN, ...
    'auxiliary_index',NaN,'population_index',NaN, ...
    'raw_xu_1',NaN,'raw_xu_2',NaN,'actual_xu_1',NaN,'actual_xu_2',NaN, ...
    'true_xu_1',NaN,'true_xu_2',NaN,'true_xl_1',NaN,'true_xl_2',NaN, ...
    'true_xl_3',NaN,'cma_update_xu_1',NaN,'cma_update_xu_2',NaN, ...
    'cma_update_xl_1',NaN,'cma_update_xl_2',NaN,'cma_update_xl_3',NaN, ...
    'cma_update_UF',NaN,'cma_update_position_changed',NaN, ...
    'boundary_repair_count',NaN, ...
    'raw_distance_to_prediction',NaN,'actual_distance_to_prediction',NaN, ...
    'relative_cma_distance',NaN,'standardized_distance',NaN, ...
    'auxiliary_UF',NaN,'direct_true_UF',NaN,'delta_UF_to_direct',NaN, ...
    'better_than_direct',NaN,'normal_candidate_count',NaN, ...
    'normal_UF_max',NaN,'normal_UF_median',NaN, ...
    'better_than_normal_pair_count',NaN,'better_than_normal_pair_rate',NaN, ...
    'better_than_all_normal',NaN,'parent_selection_applicable',NaN, ...
    'parent_selected',NaN,'history_best_UF_before_sampling',NaN, ...
    'better_than_history_best',NaN,'multi_scale_diagnosed',NaN);
end
