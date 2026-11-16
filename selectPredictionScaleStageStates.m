function [states,selection] = selectPredictionScaleStageStates(injectionLog,BI)
%SELECTPREDICTIONSCALESTAGESTATES Choose read-only diagnostic snapshots.
% The current population has already been evaluated when an injection-log
% entry is written.  This function only reads that log after the complete
% main run has ended; it never calls an objective, changes a random stream,
% or changes a CMA-ES state.

stageNames = {'early','middle','late'};
stageLower = [0,1/3,2/3];
stageUpper = [1/3,2/3,1];
stageMidpoint = [1/6,1/2,5/6];
states = repmat(emptyState(BI),0,1);
selection = repmat(emptySelection(),numel(stageNames),1);
for stageIndex = 1:numel(stageNames)
    selection(stageIndex).stage = stageNames{stageIndex};
end
if isempty(injectionLog), return; end

valid = false(numel(injectionLog),1);
UFEbefore = NaN(numel(injectionLog),1);
for index = 1:numel(injectionLog)
    entry = injectionLog(index);
    valid(index) = logical(entry.prediction_valid_used) && ...
        isfield(entry,'x_pred_actual') && all(isfinite(entry.x_pred_actual));
    UFEbefore(index) = preSamplingUFE(entry);
end
firstIndex = find(valid,1,'first');
if isempty(firstIndex), return; end

UFEstart = UFEbefore(firstIndex);
referenceBudget = BI.UmaxFEs;
denominator = referenceBudget-UFEstart;
if ~isfinite(denominator) || denominator <= 0
    return;
end
progress = (UFEbefore-UFEstart)/denominator;

for stageIndex = 1:numel(stageNames)
    inStage = valid & progress >= stageLower(stageIndex)-eps & ...
        progress <= stageUpper(stageIndex)+eps;
    if stageIndex < numel(stageNames)
        inStage = inStage & progress < stageUpper(stageIndex);
    end
    choices = find(inStage);
    if isempty(choices)
        selection(stageIndex).missing = true;
        selection(stageIndex).reason = 'no_valid_prediction_state_in_budget_stage';
        continue;
    end
    [~,relativeIndex] = min(abs(progress(choices)-stageMidpoint(stageIndex)));
    logIndex = choices(relativeIndex);
    state = makeState(injectionLog,logIndex,BI,stageNames{stageIndex}, ...
        stageIndex,progress(logIndex),UFEstart,referenceBudget);
    states(end+1,1) = state; %#ok<AGROW>
    selection(stageIndex).selected = true;
    selection(stageIndex).generation = state.generation;
    selection(stageIndex).log_index = state.log_index;
    selection(stageIndex).UFE_before_sampling = state.UFE_before_sampling;
    selection(stageIndex).progress = state.progress;
    selection(stageIndex).reason = 'selected_nearest_stage_budget_midpoint';
end
end

function value = preSamplingUFE(entry)
if isfield(entry,'UFE_before_sampling') && isfinite(entry.UFE_before_sampling)
    value = entry.UFE_before_sampling;
    return;
end
% Compatibility with older run logs that predate the pure snapshot field.
% The main population uses exactly population_size upper evaluations; an
% optional refinement may add one more.  New experiments always use the
% exact field above, so this fallback is never used for formal diagnostics.
value = entry.UFE-entry.population_size-double(entry.external_refinement_upper_evaluated);
end

function value = preSamplingLFE(entry)
if isfield(entry,'LFE_before_sampling') && isfinite(entry.LFE_before_sampling)
    value = entry.LFE_before_sampling;
else
    % Old logs do not contain an exact lower-evaluation pre-snapshot.  They
    % remain usable for optional exploratory state selection, but the new
    % formal runner always records this value directly.
    value = NaN;
end
end

function state = makeState(log,logIndex,BI,stage,stageIndex,progress, ...
        UFEstart,referenceBudget)
entry = log(logIndex);
normalMask = strcmp(entry.source_labels,'normal');
directIndex = find(strcmp(entry.source_labels,'predicted_direct'),1,'first');
assert(any(normalMask) && ~isempty(directIndex), ...
    'A scale diagnostic state requires normal candidates and one direct prediction.');
historyBest = -Inf;
if logIndex > 1
    previous = [log(1:logIndex-1).best_so_far_UF];
    previous = previous(isfinite(previous));
    if ~isempty(previous), historyBest = max(previous); end
end
state = emptyState(BI);
state.stage = stage;
state.stage_index = stageIndex;
state.generation = entry.generation;
state.log_index = logIndex;
state.progress = progress;
state.UFE_start = UFEstart;
state.reference_UFE_budget = referenceBudget;
state.UFE_before_sampling = preSamplingUFE(entry);
state.LFE_before_sampling = preSamplingLFE(entry);
state.UFE_after_population = entry.UFE;
state.LFE_after_population = entry.LFE_total;
state.x_pred = entry.x_pred_actual;
state.cma_mean = entry.cma_mean_before_sampling;
state.sigma = entry.sigma_before_sampling;
state.C = entry.C_before_sampling;
state.pc = entry.pc_before_sampling;
state.ps = entry.ps_before_sampling;
state.normal_joint = entry.joint_after_true_evaluation(normalMask,:);
state.normal_UF = entry.upper_objective_after_true_evaluation(normalMask);
state.direct_joint = entry.joint_after_true_evaluation(directIndex,:);
state.direct_UF = entry.upper_objective_after_true_evaluation(directIndex);
state.population_joint = entry.joint_after_true_evaluation;
state.population_fit = entry.population_fit(:);
state.parent_count = floor(entry.population_size/2);
state.history_best_UF_before_sampling = historyBest;
state.u_lb = BI.u_lb;
state.u_ub = BI.u_ub;
state.fn = BI.fn;
state.normal_indices = find(normalMask)';
state.direct_index = directIndex;
end

function state = emptyState(BI)
state = struct('stage','','stage_index',NaN,'generation',NaN, ...
    'log_index',NaN,'progress',NaN,'UFE_start',NaN, ...
    'reference_UFE_budget',NaN,'UFE_before_sampling',NaN, ...
    'LFE_before_sampling',NaN,'UFE_after_population',NaN, ...
    'LFE_after_population',NaN,'x_pred',NaN(1,BI.u_dim), ...
    'cma_mean',NaN(1,BI.dim),'sigma',NaN,'C',NaN(BI.dim), ...
    'pc',NaN(1,BI.dim),'ps',NaN(1,BI.dim), ...
    'normal_joint',NaN(0,BI.dim),'normal_UF',NaN(0,1), ...
    'direct_joint',NaN(1,BI.dim),'direct_UF',NaN, ...
    'population_joint',NaN(0,BI.dim),'population_fit',NaN(0,1), ...
    'parent_count',NaN,'history_best_UF_before_sampling',-Inf, ...
    'u_lb',BI.u_lb,'u_ub',BI.u_ub,'fn',BI.fn, ...
    'normal_indices',zeros(1,0),'direct_index',NaN);
end

function item = emptySelection()
item = struct('stage','','selected',false,'missing',false,'reason', ...
    'not_checked','generation',NaN,'log_index',NaN, ...
    'UFE_before_sampling',NaN,'progress',NaN);
end
