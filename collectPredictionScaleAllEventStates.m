function [states,runInfo] = collectPredictionScaleAllEventStates( ...
    injectionLog,BI,terminalUFE)
%COLLECTPREDICTIONSCALEALLEVENTSTATES Read every real F050 auxiliary event.
% This is a post-run reader only.  It never evaluates an objective, changes
% an RNG stream, or writes a CMA-ES state.  Phase labels are intentionally
% assigned after termination using the completed run's own UFE interval.

validateattributes(terminalUFE,{'numeric'},{'real','finite','scalar','nonnegative'});
states = repmat(emptyState(BI),0,1);
runInfo = struct('valid_event_count',0,'UFE_start',NaN,'UFE_end',terminalUFE, ...
    'UFE_interval',NaN,'phase_assignment_available',false, ...
    'early_event_count',0,'middle_event_count',0,'late_event_count',0, ...
    'unclassified_event_count',0);
if isempty(injectionLog), return; end

valid = false(numel(injectionLog),1);
for index = 1:numel(injectionLog)
    entry = injectionLog(index);
    valid(index) = entryIsUsableAuxiliaryEvent(entry,BI.u_dim);
end
validIndices = find(valid);
runInfo.valid_event_count = numel(validIndices);
if isempty(validIndices), return; end

UFEstart = injectionLog(validIndices(1)).UFE;
denominator = terminalUFE-UFEstart;
runInfo.UFE_start = UFEstart;
runInfo.UFE_interval = denominator;
runInfo.phase_assignment_available = denominator > 0;
for ordinal = 1:numel(validIndices)
    logIndex = validIndices(ordinal);
    entry = injectionLog(logIndex);
    if denominator > 0
        progress = (entry.UFE-UFEstart)/denominator;
        % Numerical roundoff may make a terminal event differ from one by
        % a few ulps; do not create an artificial fourth phase.
        if abs(progress-1) < 1e-12, progress = 1; end
        assert(progress >= -1e-12 && progress <= 1+1e-12, ...
            'A real auxiliary event falls outside its completed-run UFE range.');
        stage = stageForProgress(progress);
    else
        progress = NaN;
        stage = 'unclassified';
    end
    state = makeState(injectionLog,logIndex,BI,ordinal,stage,progress, ...
        UFEstart,terminalUFE);
    states(end+1,1) = state; %#ok<AGROW>
    switch stage
        case 'early', runInfo.early_event_count = runInfo.early_event_count + 1;
        case 'middle', runInfo.middle_event_count = runInfo.middle_event_count + 1;
        case 'late', runInfo.late_event_count = runInfo.late_event_count + 1;
        otherwise, runInfo.unclassified_event_count = ...
                runInfo.unclassified_event_count + 1;
    end
end
end

function valid = entryIsUsableAuxiliaryEvent(entry,upperDim)
valid = isfield(entry,'prediction_valid_used') && logical(entry.prediction_valid_used) && ...
    isfield(entry,'x_pred_actual') && numel(entry.x_pred_actual) == upperDim && ...
    all(isfinite(entry.x_pred_actual)) && isfield(entry,'source_labels') && ...
    sum(strcmp(entry.source_labels,'predicted_auxiliary')) > 0 && ...
    isfield(entry,'auxiliary_actual_joint') && ...
    size(entry.auxiliary_actual_joint,1) > 0 && ...
    isfield(entry,'upper_objective_after_true_evaluation') && ...
    isfield(entry,'UFE') && isfinite(entry.UFE);
end

function stage = stageForProgress(progress)
if progress < 1/3
    stage = 'early';
elseif progress < 2/3
    stage = 'middle';
else
    stage = 'late';
end
end

function state = makeState(log,logIndex,BI,eventOrdinal,stage,progress, ...
        UFEstart,UFEend)
entry = log(logIndex);
normalMask = strcmp(entry.source_labels,'normal');
directIndex = find(strcmp(entry.source_labels,'predicted_direct'),1,'first');
auxiliaryIndices = find(strcmp(entry.source_labels,'predicted_auxiliary'));
assert(~isempty(directIndex) && ~isempty(auxiliaryIndices), ...
    'An F050 diagnostic event needs one direct point and auxiliary candidates.');
historyBest = -Inf;
if logIndex > 1
    previous = [log(1:logIndex-1).best_so_far_UF];
    previous = previous(isfinite(previous));
    if ~isempty(previous), historyBest = max(previous); end
end

state = emptyState(BI);
state.event_ordinal = eventOrdinal;
state.stage = stage;
state.stage_index = stageIndex(stage);
state.generation = entry.generation;
state.log_index = logIndex;
state.progress = progress;
state.UFE_start = UFEstart;
state.UFE_end = UFEend;
state.UFE_before_sampling = valueOrNaN(entry,'UFE_before_sampling');
state.LFE_before_sampling = valueOrNaN(entry,'LFE_before_sampling');
state.UFE_after_population = entry.UFE;
state.LFE_after_population = entry.LFE_total;
state.x_pred = entry.x_pred_actual;
state.cma_mean = entry.cma_mean_before_sampling;
state.sigma = entry.sigma_before_sampling;
state.C = entry.C_before_sampling;
state.pc = entry.pc_before_sampling;
state.ps = entry.ps_before_sampling;
state.eta = entry.auxiliary_sampling_scale;
state.normal_joint = entry.joint_after_true_evaluation(normalMask,:);
state.normal_UF = entry.upper_objective_after_true_evaluation(normalMask);
state.direct_joint = entry.joint_after_true_evaluation(directIndex,:);
state.direct_UF = entry.upper_objective_after_true_evaluation(directIndex);
state.auxiliary_indices = auxiliaryIndices(:)';
state.auxiliary_joint = entry.joint_after_true_evaluation(auxiliaryIndices,:);
state.auxiliary_UF = entry.upper_objective_after_true_evaluation(auxiliaryIndices);
state.auxiliary_raw_joint = entry.auxiliary_raw_joint;
state.auxiliary_sampled_joint = entry.auxiliary_actual_joint;
state.auxiliary_boundary_repaired = entry.auxiliary_boundary_repaired;
state.population_joint = entry.joint_for_cma_update;
state.population_UF = entry.population_UF(:);
state.population_fit = entry.population_fit(:);
state.parent_count = floor(entry.population_size/2);
state.mean_update_applied = entry.mean_update_applied;
state.selected_parent_indices = entry.selected_parent_indices(:)';
state.history_best_UF_before_sampling = historyBest;
state.u_lb = BI.u_lb;
state.u_ub = BI.u_ub;
state.fn = BI.fn;
state.normal_indices = find(normalMask)';
state.direct_index = directIndex;
end

function value = stageIndex(stage)
switch stage
    case 'early', value = 1;
    case 'middle', value = 2;
    case 'late', value = 3;
    otherwise, value = 0;
end
end

function value = valueOrNaN(entry,name)
if isfield(entry,name) && isnumeric(entry.(name)) && isscalar(entry.(name)) && ...
        isfinite(entry.(name))
    value = entry.(name);
else
    value = NaN;
end
end

function state = emptyState(BI)
state = struct('event_ordinal',NaN,'stage','','stage_index',NaN,'generation',NaN, ...
    'log_index',NaN,'progress',NaN,'UFE_start',NaN,'UFE_end',NaN, ...
    'UFE_before_sampling',NaN,'LFE_before_sampling',NaN, ...
    'UFE_after_population',NaN,'LFE_after_population',NaN, ...
    'x_pred',NaN(1,BI.u_dim),'cma_mean',NaN(1,BI.dim),'sigma',NaN, ...
    'C',NaN(BI.dim),'pc',NaN(1,BI.dim),'ps',NaN(1,BI.dim),'eta',NaN, ...
    'normal_joint',NaN(0,BI.dim),'normal_UF',NaN(0,1), ...
    'direct_joint',NaN(1,BI.dim),'direct_UF',NaN, ...
    'auxiliary_indices',zeros(1,0),'auxiliary_joint',NaN(0,BI.dim), ...
    'auxiliary_UF',NaN(0,1),'auxiliary_raw_joint',NaN(0,BI.dim), ...
    'auxiliary_sampled_joint',NaN(0,BI.dim), ...
    'auxiliary_boundary_repaired',false(0,BI.dim), ...
    'population_joint',NaN(0,BI.dim),'population_UF',NaN(0,1), ...
    'population_fit',NaN(0,1), ...
    'parent_count',NaN,'mean_update_applied',false, ...
    'selected_parent_indices',zeros(1,0), ...
    'history_best_UF_before_sampling',-Inf,'u_lb',BI.u_lb,'u_ub',BI.u_ub, ...
    'fn',BI.fn,'normal_indices',zeros(1,0),'direct_index',NaN);
end
