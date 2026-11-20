function report = testScaleOutcomeModelExperiment(outputRoot)
%TESTSCALEOUTCOMEMODELEXPERIMENT Smoke test for one-generation intervention.
% This deliberately runs one SMD1 main trajectory and one matched branch
% repeat.  It verifies schedule timing, common-UFE alignment, and the fact
% that the small run is correctly withheld from five-fold model validation.

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_scale_outcome_model_tests', ...
        'smoke_SMD1_R1_P1');
end
experiment = runSMDScaleOutcomeModelExperiment(outputRoot,1,1,1,1);
assert(~isempty(experiment.stateSnapshots), ...
    'The smoke main run did not capture a valid quadratic-prediction state.');
outcomes = experiment.outcomes;
assert(~isempty(outcomes),'No matched intervention outcomes were recorded.');
assert(all(ismember([0.25;0.50;0.75;1.00], ...
    unique(outcomes.eta(outcomes.is_training_eta ~= 0)))), ...
    'The four discrete intervention scales were not all evaluated.');

keys = unique(string(outcomes.state_key),'stable');
for index = 1:numel(keys)
    rows = outcomes(string(outcomes.state_key) == keys(index),:);
    assert(numel(unique(rows.common_UFE_checkpoint)) == 1, ...
        'Matched branches did not use one common UFE checkpoint.');
    reference = rows(rows.is_reference_eta ~= 0,:);
    assert(height(reference) == 1 && abs(reference.relative_overall_gain) < 1e-12, ...
        'The eta=.50 reference outcome is not exactly zero.');
end

objects = experiment.branchObjects;
for index = 1:numel(objects)
    item = objects{index};
    log = item.ins.predictionInjectionLog;
    assert(numel(log) == 5, ...
        'An intervention branch did not complete its requested H=5 horizon.');
    used = [log.auxiliary_sampling_scale];
    assert(abs(used(1)-item.eta) < 1e-12 && ...
        max(abs(used(2:end)-0.50)) < 1e-12, ...
        'The test eta was not limited to the first resumed generation.');
    for generation = 1:numel(log)
        assert(numel(log(generation).candidate_upper_evaluation_UFE) == 8, ...
            'Candidate UFE checkpoints were not recorded for aligned outcomes.');
    end
end
% First resumed generation must differ only in the requested auxiliary
% scale: normal candidates and the direct prediction are snapshot-paired.
stateKeys = unique(cellfun(@(item)sprintf('%d_%d_%d',item.SMD, ...
    item.state_index,item.repeat_no),objects,'UniformOutput',false));
for keyIndex = 1:numel(stateKeys)
    members = objects(cellfun(@(item)strcmp(sprintf('%d_%d_%d',item.SMD, ...
        item.state_index,item.repeat_no),stateKeys{keyIndex}),objects));
    eta025 = members(cellfun(@(item)abs(item.eta-0.25) < 10*eps,members));
    eta050 = members(cellfun(@(item)abs(item.eta-0.50) < 10*eps,members));
    assert(numel(eta025) == 1 && numel(eta050) == 1, ...
        'Missing paired eta=.25/.50 branch objects.');
    log025 = eta025{1}.ins.predictionInjectionLog(1);
    log050 = eta050{1}.ins.predictionInjectionLog(1);
    normal025 = strcmp(log025.source_labels,'normal');
    normal050 = strcmp(log050.source_labels,'normal');
    assert(isequal(normal025,normal050) && isequal( ...
        log025.joint_after_injection(normal025,:), ...
        log050.joint_after_injection(normal050,:)), ...
        'First-generation normal candidates are not paired across eta.');
    direct025 = find(strcmp(log025.source_labels,'predicted_direct'));
    direct050 = find(strcmp(log050.source_labels,'predicted_direct'));
    if ~isempty(direct025) && ~isempty(direct050)
        assert(isequal(log025.joint_after_injection(direct025,:), ...
            log050.joint_after_injection(direct050,:)), ...
            'The direct prediction is not paired across eta.');
    end
    if log025.prediction_valid_used && log050.prediction_valid_used
        assert(max(abs(log050.auxiliary_upper_raw_offsets(:)- ...
            2*log025.auxiliary_upper_raw_offsets(:))) < 1e-9, ...
            'The eta=.25/.50 branches do not share auxiliary base directions.');
    end
end
assert(~isempty(experiment.modelResult.parameterTable) && ...
    all(strcmp(experiment.modelResult.parameterTable.status, ...
    'insufficient_independent_main_runs_for_grouped_fivefold_cv')), ...
    'One independent main run must not be presented as a valid grouped-CV model.');
% Snapshot collection and all later branches must leave the primary run
% exactly reproducible under its original seed and counters.
mainSaved = load(fullfile(outputRoot,'SMD1_run01_main.mat'), ...
    'mainIns','mainRngAfter','BI');
global ulFunctionEvaluations llFunctionEvaluations;
savedRng = rng; savedUFE = ulFunctionEvaluations; savedLFE = llFunctionEvaluations;
cleanup = onCleanup(@() restoreOuterState(savedRng,savedUFE,savedLFE)); %#ok<NASGU>
rng(mainSaved.BI.predictionInjection.randomSeed,'twister');
reference = BLCMAES(mainSaved.BI);
assert(isequaln([reference.UF,reference.LF,reference.UFEs,reference.LFEs], ...
    [mainSaved.mainIns.UF,mainSaved.mainIns.LF,mainSaved.mainIns.UFEs, ...
    mainSaved.mainIns.LFEs]) && isequaln(reference.record,mainSaved.mainIns.record) && ...
    isequal(mainSaved.mainRngAfter,rng), ...
    'Offline scale branches or snapshots changed the main BL-CMA-ES path.');
report = struct('outputRoot',outputRoot,'stateCount',numel(keys), ...
    'outcomeRows',height(outcomes),'passed',true);
save(fullfile(outputRoot,'scale_outcome_model_smoke_report.mat'),'report');
fprintf('Scale-outcome model smoke test passed: %s\n',outputRoot);
end

function restoreOuterState(savedRng,savedUFE,savedLFE)
global ulFunctionEvaluations llFunctionEvaluations;
rng(savedRng);
ulFunctionEvaluations = savedUFE;
llFunctionEvaluations = savedLFE;
end
