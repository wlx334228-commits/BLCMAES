function report = testNeuralPredictionEndToEnd(outputRoot)
%TESTNEURALPREDICTIONENDTOEND Small real SMD1 pipeline verification.
% The run is intentionally short and is not part of the formal 20-run
% analysis.  It confirms that a NN prediction, if valid, is inserted before
% lower search and that the evaluated coordinates are the update coordinates.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_neural_prediction_tests');
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end
BI = getBLOPinfo('SMD',1,5);
BI.UmaxFEs = 128;
BI.UmaxImprFEs = 128;
BI.LmaxFEs = 80;
BI.LmaxImprFEs = 20;
seed = 917;
BI.enablePredictionCorrection = false;
BI.predictionCorrection = struct('logEnabled',false);
BI.enableIndividualPredictionCorrection = false;
BI.individualPredictionCorrection = struct('mode','A0','logEnabled',false);
BI.predictionInjection = struct('mode','K1','predictionInjectCount',1, ...
    'candidatePolicy','direct_plus_auxiliary','predictionModelType','neural_mlp', ...
    'logEnabled',true,'startGeneration',10,'historyGenerations',10, ...
    'sampleCount',40,'randomSeed',seed,'auxiliaryRandomSeed',500000+seed, ...
    'neuralRandomSeed',700000+seed,'consistencyCheckEnabled',true);
rng(seed,'twister');
ins = BLCMAES(BI);
log = ins.predictionInjectionLog;
assert(~isempty(log),'The end-to-end smoke run produced no injection log.');
for generation = 1:numel(log)
    entry = log(generation);
    assert(size(entry.upper_entering_lower_search,1) == 8 && ...
        size(entry.joint_for_cma_update,1) == 8, ...
        'A candidate population unexpectedly changed size.');
    if entry.prediction_valid_used && isfinite(entry.direct_index)
        directIndex = entry.direct_index;
        assert(norm(entry.upper_entering_lower_search(directIndex,:) - ...
            entry.x_pred_actual) <= 1e-10, ...
            'The direct NN point did not enter lower-level search unchanged.');
        assert(norm(entry.joint_after_true_evaluation(directIndex,1:BI.u_dim) - ...
            entry.x_pred_actual) <= 1e-10, ...
            'The direct NN point was not the point given a true evaluation.');
    end
end
report = struct('passed',true,'seed',seed,'UFE',ins.UFEs,'LFE',ins.LFEs, ...
    'generations',numel(log),'prediction_valid_generations', ...
    sum([log.prediction_valid_used]),'injectionLog',log, ...
    'notes',['This small genuine SMD1 run is a pipeline test only; it is ' ...
     'not included in the paired formal experiment statistics.']);
if report.prediction_valid_generations == 0
    warning('BLCMAES:NeuralPredictionSmokeNoInjection', ...
        ['The small run reached no valid NN injection; the structural ' ...
         'pipeline checks still passed. Inspect the saved log before using ' ...
         'this result as a prediction-quality check.']);
end
save(fullfile(outputRoot,'neural_prediction_end_to_end_test_report.mat'), ...
    'report','-v7');
fprintf('Neural prediction end-to-end test passed: %s\n',outputRoot);
end
