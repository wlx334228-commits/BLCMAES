function experiment = runSMDPredictionRLAdaptiveSampling( ...
    outputRoot,numberOfRuns,problemNumbers)
%RUNSMDPREDICTIONRLADAPTIVESAMPLING Paired 2-D online RL auxiliary study.
% A0   : 8 normal CMA-ES candidates.
% Q    : 7 normal candidates + 1 direct quadratic prediction.
% F050 : 5 normal + 1 direct + 2 CMA-shape auxiliaries, fixed eta=0.50.
% RL1  : F050 layout; REINFORCE learns eta, delta=0.
% RL2  : F050 layout; eta=0.50, REINFORCE learns delta.
% RL3  : F050 layout; REINFORCE jointly learns eta and delta.
%
% The only RL-controlled object is the auxiliary upper covariance.  The
% original CMA-ES mean, sigma, C, paths, population size, lower solver and
% stopping rules are identical to the existing prediction-injection study.
%
% Formal experiment:
%   runSMDPredictionRLAdaptiveSampling
% Smoke batch:
%   runSMDPredictionRLAdaptiveSampling([],1,1)

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_prediction_rl_adaptive_sampling_2D', ...
        sprintf('SMD%d_to_SMD%d_%druns',min(problemNumbers), ...
        max(problemNumbers),numberOfRuns));
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

specifications = rlSpecifications(outputRoot);
saveOrValidateRLConfig(outputRoot,numberOfRuns,problemNumbers,specifications);
smokeRoot = fullfile(outputRoot,'smoke_test_RL');
if exist(fullfile(smokeRoot,'prediction_rl_adaptive_sampling_test_report.mat'), ...
        'file') ~= 2
    testPredictionRLAdaptiveSampling(smokeRoot);
else
    fprintf('Reuse completed RL auxiliary sampling smoke test: %s\n',smokeRoot);
end

for index = 1:numel(specifications)
    spec = specifications(index);
    fprintf('Run %s: %d normal + %d direct + %d auxiliary; %s.\n', ...
        spec.id,spec.normalCount,spec.directCount,spec.auxiliaryCount, ...
        spec.description);
    runSMDPredictionInjection(spec.resultRoot,numberOfRuns,problemNumbers, ...
        spec.injectCount,'direct_plus_auxiliary',5,40, ...
        spec.distribution,spec.fixedScale,'quadratic',spec.overrides);
end

runs = collectRLRuns(specifications,problemNumbers,numberOfRuns);
[summary,pairedTests] = summarisePredictionRLAdaptiveSampling( ...
    runs,specifications,problemNumbers);
writetable(runs,fullfile(outputRoot,'prediction_rl_runs.csv'));
writetable(summary,fullfile(outputRoot,'prediction_rl_summary.csv'));
writetable(pairedTests,fullfile(outputRoot, ...
    'prediction_rl_paired_wilcoxon.csv'));
experiment = struct('outputRoot',outputRoot,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'runs',runs,'summary',summary,'pairedTests',pairedTests, ...
    'notes',['All modes use the same seeds and quadratic predictor. ' ...
        'RL actions use an independent per-run stream and never consume ' ...
        'the original CMA-ES/lower-search random stream.']);
save(fullfile(outputRoot,'prediction_rl_experiment_index.mat'), ...
    'experiment','-v7');
fprintf('\nOnline RL auxiliary-sampling experiment completed: %s\n',outputRoot);
end

function specifications = rlSpecifications(outputRoot)
base = struct('rlFixedEta',0.50,'rlEtaMin',0.15,'rlEtaMax',1.25, ...
    'rlDeltaMax',log(2),'rlHiddenUnits',8,'rlLearningRate',0.01, ...
    'rlActorL2',1e-4,'rlInitialActionStd',0.25, ...
    'rlMinActionStd',0.08,'rlMaxActionStd',0.60, ...
    'rlRewardBaselineRate',0.10,'rlRewardClip',3.0, ...
    'rlGradientClip',5.0,'rlCovarianceEigenvalueFloor',1e-10, ...
    'rlCovarianceMaxCondition',1e4);
specifications = [ ...
    makeSpec('A0',0,8,0,0,'cma_gaussian',1.0,struct(), ...
        'original BL-CMA-ES',outputRoot); ...
    makeSpec('Q',1,7,1,0,'cma_gaussian',1.0,struct(), ...
        'direct quadratic prediction only',outputRoot); ...
    makeSpec('F050',3,5,1,2,'cma_gaussian',0.50,struct(), ...
        'fixed eta=0.50 CMA-shape auxiliary Gaussian',outputRoot); ...
    makeSpec('RL1',3,5,1,2,'rl_adaptive_gaussian',1.0, ...
        mergeStruct(base,struct('rlEnabled',true,'rlMode','eta_only')), ...
        'REINFORCE eta only; delta=0',outputRoot); ...
    makeSpec('RL2',3,5,1,2,'rl_adaptive_gaussian',1.0, ...
        mergeStruct(base,struct('rlEnabled',true,'rlMode','delta_only')), ...
        'REINFORCE delta only; fixed eta=0.50',outputRoot); ...
    makeSpec('RL3',3,5,1,2,'rl_adaptive_gaussian',1.0, ...
        mergeStruct(base,struct('rlEnabled',true,'rlMode','joint')), ...
        'joint REINFORCE eta and delta',outputRoot)];
end

function spec = makeSpec(id,K,normalCount,directCount,auxiliaryCount, ...
    distribution,fixedScale,overrides,description,outputRoot)
assert(normalCount+directCount+auxiliaryCount == 8, ...
    'Each RL-study population must contain exactly eight candidates.');
spec = struct('id',id,'injectCount',K,'normalCount',normalCount, ...
    'directCount',directCount,'auxiliaryCount',auxiliaryCount, ...
    'distribution',distribution,'fixedScale',fixedScale, ...
    'overrides',overrides,'description',description, ...
    'resultRoot',fullfile(outputRoot,[id '_' distribution]));
end

function saveOrValidateRLConfig(outputRoot,numberOfRuns,problemNumbers,specifications)
fileName = fullfile(outputRoot,'prediction_rl_experiment_config.mat');
config = struct('totalDim',5,'upperDim',2,'lowerDim',3, ...
    'populationSize',8,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'sampleCount',40,'startGeneration',10, ...
    'historyGenerations',10,'specifications',serialisableSpecs(specifications), ...
    'comparisons',{{'Q_vs_A0','F050_vs_Q','RL1_vs_F050', ...
        'RL2_vs_F050','RL3_vs_F050','RL3_vs_Q'}}, ...
    'notes',['One shared action controls two auxiliary samples. The Actor ' ...
        'is reset for every run and is trained only after true evaluation.']);
if exist(fileName,'file') ~= 2
    save(fileName,'config');
    return;
end
loaded = load(fileName,'config'); previous = loaded.config;
assert(previous.totalDim == 5 && previous.numberOfRuns == numberOfRuns && ...
    isequal(previous.problemNumbers,problemNumbers) && ...
    isequaln(previous.specifications,serialisableSpecs(specifications)), ...
    ['This output directory has a different RL experiment configuration. ' ...
     'Choose a fresh outputRoot rather than mixing experiments.']);
end

function stored = serialisableSpecs(specifications)
stored = specifications;
for i = 1:numel(stored)
    stored(i).resultRoot = [stored(i).id '_' stored(i).distribution];
end
end

function allRuns = collectRLRuns(specifications,problemNumbers,numberOfRuns)
allRuns = table();
for index = 1:numel(specifications)
    spec = specifications(index);
    csvFile = fullfile(spec.resultRoot,'injection_runs.csv');
    assert(exist(csvFile,'file') == 2,'Missing result table: %s',csvFile);
    rows = readtable(csvFile);
    rows = rows(ismember(rows.SMD,problemNumbers) & ...
        rows.prediction_inject_count == spec.injectCount,:);
    validateSchedule(rows,spec,problemNumbers,numberOfRuns);
    diagnostics = collectRLDiagnostics(rows,spec.resultRoot);
    rows = [rows diagnostics];
    rows.experiment_id = repmat({spec.id},height(rows),1);
    rows.sampling_distribution = repmat({spec.distribution},height(rows),1);
    rows.configured_normal_candidate_count = repmat(spec.normalCount,height(rows),1);
    rows.configured_direct_candidate_count = repmat(spec.directCount,height(rows),1);
    rows.configured_auxiliary_candidate_count = repmat(spec.auxiliaryCount,height(rows),1);
    rows.result_root = repmat({[spec.id '_' spec.distribution]},height(rows),1);
    if isempty(allRuns), allRuns = rows; else, allRuns = [allRuns;rows]; end %#ok<AGROW>
end
allRuns = sortrows(allRuns,{'SMD','experiment_id','run_no'});
end

function diagnostics = collectRLDiagnostics(rows,resultRoot)
names = {'rl_action_generations','rl_actor_update_generations', ...
    'rl_eta_median','rl_eta_IQR','rl_delta_median','rl_delta_IQR', ...
    'rl_reward_median','rl_reward_IQR','rl_advantage_median', ...
    'rl_trace_ratio_median','rl_covariance_condition_median', ...
    'rl_auxiliary_UF_median','rl_direct_true_UF_median', ...
    'rl_auxiliary_parent_selection_rate','rl_direct_parent_selection_rate', ...
    'rl_normal_parent_selection_rate','rl_final_actor_parameter_norm'};
diagnostics = array2table(NaN(height(rows),numel(names)),'VariableNames',names);
for i = 1:height(rows)
    matPath = fullfile(resultRoot,portablePath(tableText(rows.mat_file,i)));
    loaded = load(matPath,'injectionLog'); log = loaded.injectionLog;
    action = [log.rl_action_valid]'; updated = [log.rl_actor_updated]';
    eta = [log.rl_eta]'; delta = [log.rl_delta]'; reward = [log.rl_reward]';
    advantage = [log.rl_advantage]'; traceRatio = [log.rl_covariance_trace_ratio]';
    condition = [log.rl_covariance_condition]'; directUF = [log.direct_true_UF]';
    auxUF = [];
    for g = 1:numel(log), auxUF = [auxUF;log(g).rl_auxiliary_UF(:)]; end %#ok<AGROW>
    [diagnostics.rl_eta_median(i),diagnostics.rl_eta_IQR(i)] = medianIQR(eta(action));
    [diagnostics.rl_delta_median(i),diagnostics.rl_delta_IQR(i)] = medianIQR(delta(action));
    [diagnostics.rl_reward_median(i),diagnostics.rl_reward_IQR(i)] = medianIQR(reward(updated));
    diagnostics.rl_action_generations(i) = sum(action);
    diagnostics.rl_actor_update_generations(i) = sum(updated);
    diagnostics.rl_advantage_median(i) = finiteMedian(advantage(updated));
    diagnostics.rl_trace_ratio_median(i) = finiteMedian(traceRatio(action));
    diagnostics.rl_covariance_condition_median(i) = finiteMedian(condition(action));
    diagnostics.rl_auxiliary_UF_median(i) = finiteMedian(auxUF);
    diagnostics.rl_direct_true_UF_median(i) = finiteMedian(directUF);
    diagnostics.rl_auxiliary_parent_selection_rate(i) = ...
        safeRatio(sum([log.auxiliary_parent_count]), ...
        2*sum(action));
    diagnostics.rl_direct_parent_selection_rate(i) = ...
        safeRatio(sum([log.direct_parent_count]),sum(action));
    diagnostics.rl_normal_parent_selection_rate(i) = ...
        safeRatio(sum([log.normal_parent_count]),sum([log.normal_sample_count]));
    finalNorm = [log.rl_actor_parameter_norm_after]';
    diagnostics.rl_final_actor_parameter_norm(i) = finiteLast(finalNorm);
end
end

function validateSchedule(rows,spec,problemNumbers,numberOfRuns)
for smdNo = problemNumbers(:)'
    subset = rows(rows.SMD == smdNo,:);
    assert(height(subset) == numberOfRuns && ...
        isequal(sort(subset.run_no)',1:numberOfRuns), ...
        '%s has incomplete or duplicate SMD%d runs.',spec.id,smdNo);
    expectedSeed = (smdNo-1)*numberOfRuns+subset.run_no;
    assert(all(subset.seed == expectedSeed), ...
        '%s does not use the paired seed schedule for SMD%d.',spec.id,smdNo);
end
end

function result = mergeStruct(base,extra)
result = base; names = fieldnames(extra);
for i = 1:numel(names), result.(names{i}) = extra.(names{i}); end
end

function path = portablePath(path)
path = strrep(path,'/',filesep); path = strrep(path,'\\',filesep);
end

function value = tableText(column,index)
if iscell(column), value = column{index}; else, value = char(column(index)); end
end

function value = safeRatio(numerator,denominator)
if denominator <= 0, value = NaN; else, value = numerator/denominator; end
end

function value = finiteMedian(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = median(values); end
end

function value = finiteLast(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = values(end); end
end

function [middle,spread] = medianIQR(values)
values = sort(values(isfinite(values)));
if isempty(values), middle = NaN; spread = NaN; return; end
middle = median(values);
if numel(values) < 2, spread = NaN; return; end
spread = percentile(values,75)-percentile(values,25);
end

function value = percentile(values,p)
position = 1+(numel(values)-1)*p/100; lo = floor(position); hi = ceil(position);
if lo == hi, value = values(lo); else, value = values(lo)+(position-lo)*(values(hi)-values(lo)); end
end
