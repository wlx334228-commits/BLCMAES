function [rows,diagnosticTotals] = evaluatePredictionScaleDiagnosticState( ...
    state,BI,smdNo,runNo,runSeed,etas)
%EVALUATEPREDICTIONSCALEDIAGNOSTICSTATE Offline paired scale evaluation.
% A state has already occurred in a completed main run.  The diagnostic
% candidates below are never appended to its archive/population. Their two
% upper directions are recovered from this event's logged raw F050 samples,
% so all eta values use exactly the same directions. Global counters are
% temporarily used only because the original SMD evaluators maintain them;
% their caller-visible values are restored before return.
% The original eta=0.5 UF is retained in each output row for audit, but is
% re-evaluated here under the same private lower-search stream as the other
% eta values. This avoids treating lower-level random variation as a scale
% effect in the paired comparison.

validateattributes(etas,{'numeric'},{'vector','real','finite','positive'});
assert(BI.u_dim == 2 && BI.l_dim == 3, ...
    'This first diagnostic module is intentionally limited to DU=2, DL=3.');
assert(any(abs(etas-state.eta) < 1e-12), ...
    'The diagnostic eta set must include the logged main auxiliary scale.');
[z,directionInfo] = recoverPredictionScaleDiagnosticDirections(state);
candidates = generatePredictionScaleDiagnosticCandidates(state,etas,z);
template = emptyCandidateRow();
rows = repmat(template,numel(candidates),1);

global ulFunctionEvaluations llFunctionEvaluations;
savedUFE = ulFunctionEvaluations;
savedLFE = llFunctionEvaluations;
ulFunctionEvaluations = 0;
llFunctionEvaluations = 0;
restoreCounters = onCleanup(@() restoreGlobalCounters(savedUFE,savedLFE)); %#ok<NASGU>
for rowIndex = 1:numel(candidates)
    sample = candidates(rowIndex);
    % Reset the lower-search private stream for the same candidate index at
    % every eta.  Thus eta affects only the upper displacement magnitude;
    % lower-level stochasticity is paired as well.
    lowerSeed = makeSeed(smdNo,runNo,state.generation, ...
        100+sample.candidate_index);
    lowerStream = RandStream('mt19937ar','Seed',lowerSeed);
    evaluation = evaluatePredictionScaleDiagnosticCandidate( ...
        sample.actual_upper,state,BI,lowerStream);
    rows(rowIndex) = makeCandidateRow(state,BI,smdNo,runNo,runSeed, ...
        lowerSeed,directionInfo,sample,evaluation);
end
diagnosticTotals = struct('UFE',ulFunctionEvaluations, ...
    'LFE',llFunctionEvaluations,'candidate_count',numel(rows), ...
    'direction_source',directionInfo.source, ...
    'direction_reconstruction_relative_residual', ...
        directionInfo.reconstruction_relative_residual);
end

function row = makeCandidateRow(state,BI,smdNo,runNo,runSeed,lowerSeed, ...
        directionInfo,sample,evaluation)
normalUF = state.normal_UF(:)';
normalPairRate = mean(evaluation.UF > normalUF);
candidateFit = evaluation.UF; % SMD1/3/6/8 are unconstrained in this study.
virtualRank = 1+sum(state.population_fit > candidateFit);
row = emptyCandidateRow();
row.SMD = smdNo;
row.run_no = runNo;
row.seed = runSeed;
row.state_id = stateIdentifier(smdNo,runNo,state);
row.stage = state.stage;
row.stage_index = state.stage_index;
row.generation = state.generation;
row.log_index = state.log_index;
row.progress = state.progress;
row.UFE_start = state.UFE_start;
row.reference_UFE_budget = stateUFEEnd(state);
row.UFE_end = stateUFEEnd(state);
row.main_UFE_before_sampling = state.UFE_before_sampling;
row.main_LFE_before_sampling = state.LFE_before_sampling;
row.main_UFE_after_population = state.UFE_after_population;
row.main_LFE_after_population = state.LFE_after_population;
row.x_pred_1 = state.x_pred(1); row.x_pred_2 = state.x_pred(2);
row.cma_mean_1 = state.cma_mean(1); row.cma_mean_2 = state.cma_mean(2);
row.sigma = state.sigma;
row.Cuu_11 = state.C(1,1); row.Cuu_12 = state.C(1,2);
row.Cuu_22 = state.C(2,2);
row.eta = sample.eta;
row.candidate_index = sample.candidate_index;
row.z_1 = sample.z(1); row.z_2 = sample.z(2);
row.direction_source = directionInfo.source;
row.direction_reconstruction_relative_residual = ...
    directionInfo.reconstruction_relative_residual;
row.z_seed = NaN; row.lower_random_seed = lowerSeed;
row.raw_xu_1 = sample.raw_upper(1); row.raw_xu_2 = sample.raw_upper(2);
row.actual_xu_1 = sample.actual_upper(1); row.actual_xu_2 = sample.actual_upper(2);
row.main_logged_eta = state.eta;
row.main_logged_auxiliary_UF = state.auxiliary_UF(sample.candidate_index);
loggedRaw = state.auxiliary_raw_joint(sample.candidate_index,1:BI.u_dim);
row.raw_upper_matches_logged_main = double( ...
    norm(sample.raw_upper-loggedRaw) <= 1e-10*max(1,norm(loggedRaw)));
row.boundary_repaired_1 = sample.boundary_repaired(1);
row.boundary_repaired_2 = sample.boundary_repaired(2);
row.boundary_repair_count = sum(sample.boundary_repaired);
row.raw_distance_to_prediction = sample.raw_distance;
row.actual_distance_to_prediction = sample.actual_distance;
row.relative_cma_distance = sample.relative_cma_distance;
row.standardized_distance = sample.standardized_distance;
row.UF = evaluation.UF; row.LF = evaluation.LF;
row.upper_error = abs(evaluation.UF-BI.u_fopt);
row.lower_error = abs(evaluation.LF-BI.l_fopt);
row.diagnostic_UFE = evaluation.diagnostic_UFE;
row.diagnostic_LFE = evaluation.diagnostic_LFE;
row.diagnostic_evaluation_valid = all(isfinite([evaluation.UF,evaluation.LF, ...
    evaluation.UC,evaluation.LC]));
row.diagnostic_invalid_evaluation = ~row.diagnostic_evaluation_valid;
row.direct_true_UF = state.direct_UF;
row.delta_UF_to_direct = evaluation.UF-state.direct_UF;
row.better_than_direct = evaluation.UF > state.direct_UF;
row.normal_candidate_count = numel(normalUF);
row.normal_UF_max = max(normalUF);
row.normal_UF_median = median(normalUF);
row.better_than_normal_pair_count = sum(evaluation.UF > normalUF);
row.better_than_normal_pair_rate = normalPairRate;
row.better_than_all_normal = all(evaluation.UF > normalUF);
row.virtual_parent_rank = virtualRank;
row.virtual_enters_parent = virtualRank <= state.parent_count;
row.history_best_UF_before_sampling = state.history_best_UF_before_sampling;
row.better_than_history_best = evaluation.UF > state.history_best_UF_before_sampling;
end

function restoreGlobalCounters(UFE,LFE)
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations = UFE;
llFunctionEvaluations = LFE;
end

function seed = makeSeed(smdNo,runNo,generation,salt)
seed = mod(100000+10000*smdNo+1000*runNo+10*generation+salt, ...
    2^31-2)+1;
end

function id = stateIdentifier(smdNo,runNo,state)
% Phase is a post-run analysis label. The same real generation must retain
% one stable identifier even if a later analysis changes the phase scheme.
id = sprintf('SMD%d_run%02d_gen%04d',smdNo,runNo,state.generation);
end

function row = emptyCandidateRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_id','', ...
    'stage','','stage_index',NaN,'generation',NaN,'log_index',NaN, ...
    'progress',NaN,'UFE_start',NaN,'reference_UFE_budget',NaN,'UFE_end',NaN, ...
    'main_UFE_before_sampling',NaN,'main_LFE_before_sampling',NaN, ...
    'main_UFE_after_population',NaN,'main_LFE_after_population',NaN, ...
    'x_pred_1',NaN,'x_pred_2',NaN,'cma_mean_1',NaN,'cma_mean_2',NaN, ...
    'sigma',NaN,'Cuu_11',NaN,'Cuu_12',NaN,'Cuu_22',NaN, ...
    'eta',NaN,'candidate_index',NaN,'z_1',NaN,'z_2',NaN, ...
    'direction_source','','direction_reconstruction_relative_residual',NaN, ...
    'z_seed',NaN,'lower_random_seed',NaN,'raw_xu_1',NaN,'raw_xu_2',NaN, ...
    'actual_xu_1',NaN,'actual_xu_2',NaN,'boundary_repaired_1',false, ...
    'boundary_repaired_2',false,'boundary_repair_count',NaN, ...
    'raw_distance_to_prediction',NaN,'actual_distance_to_prediction',NaN, ...
    'relative_cma_distance',NaN,'standardized_distance',NaN, ...
    'UF',NaN,'LF',NaN,'upper_error',NaN,'lower_error',NaN, ...
    'diagnostic_UFE',NaN,'diagnostic_LFE',NaN, ...
    'diagnostic_evaluation_valid',false,'diagnostic_invalid_evaluation',false, ...
    'main_logged_eta',NaN,'main_logged_auxiliary_UF',NaN, ...
    'raw_upper_matches_logged_main',false,'direct_true_UF',NaN, ...
    'delta_UF_to_direct',NaN,'better_than_direct',false, ...
    'normal_candidate_count',NaN,'normal_UF_max',NaN, ...
    'normal_UF_median',NaN,'better_than_normal_pair_count',NaN, ...
    'better_than_normal_pair_rate',NaN,'better_than_all_normal',false, ...
    'virtual_parent_rank',NaN,'virtual_enters_parent',false, ...
    'history_best_UF_before_sampling',NaN,'better_than_history_best',false);
end

function value = stateUFEEnd(state)
% New all-event records use the actual terminal UFE.  Retain the old field
% name as a backwards-compatible CSV column for the representative-state
% diagnostic, whose historical logs used a common reference budget.
if isfield(state,'UFE_end') && isfinite(state.UFE_end)
    value = state.UFE_end;
elseif isfield(state,'reference_UFE_budget')
    value = state.reference_UFE_budget;
else
    value = NaN;
end
end
