function report = testNextGenerationScaleExperiment(outputRoot)
%TESTNEXTGENERATIONSCALEEXPERIMENT Smoke test for all-state t+1 probing.
% It verifies candidate composition, paired first-generation normal draws,
% five normal-only t+1 probe evaluations, and isolation of the reference
% main trajectory. It intentionally uses one SMD1 main run and one repeat;
% this is a correctness test, not a model-validation experiment.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_nextgen_scale_tests','smoke_SMD1_R1');
end
experiment = runSMDNextGenerationScaleExperiment(outputRoot,1,1,1);
outcomes = experiment.outcomeTable;
updates = experiment.updateTable;
candidates = experiment.candidateTable;
completed = outcomes(outcomes.probe_completed ~= 0,:);
assert(~isempty(completed),'Smoke test did not reach any completed t+1 normal probe.');
k3 = completed(strcmp(completed.candidate_policy,'K3'),:);
a1 = completed(strcmp(completed.candidate_policy,'A1'),:);
assert(all(k3.intervention_normal_count == 5 & k3.intervention_direct_count == 1 & ...
    k3.intervention_auxiliary_count == 2),'K3 layout is not 5 normal + 1 direct + 2 auxiliary.');
assert(all(k3.probe_normal_count == 5),'K3 did not evaluate five normal-only probe candidates.');
assert(all(a1.intervention_normal_count == 7 & a1.intervention_direct_count == 1 & ...
    a1.intervention_auxiliary_count == 0),'A1 layout is not 7 normal + 1 direct.');

states = unique(k3(:,{'state_index','repeat_no'}),'rows');
for i = 1:height(states)
    sub = k3(k3.state_index == states.state_index(i) & ...
        k3.repeat_no == states.repeat_no(i),:);
    assert(height(sub) == 4 && all(abs(sort(sub.eta)-[.25;.5;.75;1]) < 1e-12), ...
        'A completed K3 diagnostic state lacks a matched eta quartet.');
    detail = updates(updates.state_index == states.state_index(i) & ...
        updates.repeat_no == states.repeat_no(i) & strcmp(updates.candidate_policy,'K3'),:);
    assert(height(detail) == 4 && numel(unique(detail.upper_before_injection)) == 1, ...
        'The four eta branches did not reuse the same current-generation normal draw.');
    probe = candidates(candidates.state_index == states.state_index(i) & ...
        candidates.repeat_no == states.repeat_no(i) & strcmp(candidates.candidate_policy,'K3'),:);
    referenceDraws = probe.standard_normal_draw(strcmp(probe.variant,'eta_050'));
    for eta = [.25 .75 1]
        compared = probe.standard_normal_draw(abs(probe.eta-eta) < 1e-12);
        assert(isequal(referenceDraws,compared), ...
            'The t+1 normal-probe branches did not reuse the same standard-normal directions.');
    end
end
assert(all(candidates.candidate_index >= 1 & candidates.candidate_index <= 5), ...
    'Probe candidate table has a non-normal or out-of-range candidate index.');
assert(all(isfinite(candidates.UF) & isfinite(candidates.LF)), ...
    'A true t+1 normal probe evaluation is non-finite.');

% Snapshot capture and all branch work must leave the reference trajectory
% untouched. Re-run the reference once without branch diagnostics.
BI = getBLOPinfo('SMD',1,5);
BI.UmaxFEs=2500; BI.UmaxImprFEs=350; BI.LmaxFEs=250; BI.LmaxImprFEs=25;
BI.enablePredictionCorrection=false; BI.predictionCorrection=struct('logEnabled',false);
BI.enableIndividualPredictionCorrection=false;
BI.individualPredictionCorrection=struct('mode','A0','logEnabled',false);
BI.predictionInjection=struct('mode','K3','predictionInjectCount',3, ...
    'candidatePolicy','direct_plus_auxiliary','auxiliarySamplingDistribution','cma_gaussian', ...
    'auxiliarySamplingScale',.5,'predictionModelType','quadratic','logEnabled',true, ...
    'startGeneration',10,'historyGenerations',10,'useAllHistory',false,'sampleCount',40, ...
    'maxPredictionNormalizedDistance',Inf,'auxiliaryRandomSeed',500001, ...
    'randomSeed',1,'consistencyCheckEnabled',true);
rng(1,'twister'); baseline = BLCMAES(BI);
saved = load(fullfile(outputRoot,'SMD1_run001_main.mat'),'mainIns');
assert(baseline.UFEs == saved.mainIns.UFEs && baseline.LFEs == saved.mainIns.LFEs && ...
    abs(baseline.UF-saved.mainIns.UF) < 1e-12 && ...
    norm(baseline.UX-saved.mainIns.UX) < 1e-12, ...
    'Capture/branch diagnostics changed the reference main trajectory.');

report = struct('outputRoot',outputRoot,'completedK3States',height(k3)/4, ...
    'completedA1States',height(a1),'candidateRows',height(candidates), ...
    'mainTrajectoryIsolated',true);
save(fullfile(outputRoot,'nextgen_scale_smoke_report.mat'),'report');
fprintf('Next-generation scale smoke test passed: %s\n',outputRoot);
end
