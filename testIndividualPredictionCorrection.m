function report = testIndividualPredictionCorrection(outputDir)
%TESTINDIVIDUALPREDICTIONCORRECTION Verify v2 pointwise sample correction.
% It runs a synthetic direction test and matched SMD1 integration pair.
projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_individual_prediction_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
assert(exist(outputDir,'dir') ~= 7, ...
    'Use a new output directory; test data will not be overwritten.');
mkdir(outputDir);

BI = getBLOPinfo('SMD',1,5);
BI.enableIndividualPredictionCorrection = true;
BI.individualPredictionCorrection = struct('weight',0.1,'startGeneration',10);
options = individualPredictionCorrectionOptions(BI);
CMA = struct('xmean',[0,0,1,2,3],'sigma',0.5, ...
    'C',eye(5),'pc',ones(1,5),'ps',2*ones(1,5), ...
    'B',eye(5),'D',ones(1,5),'invsqrtC',eye(5));
state = initializeIndividualPredictionState(BI);
state.pending = struct('valid',true,'x_pred',[0.5,-0.5], ...
    'source_generation',10,'model',struct());
rawUpper = [1,-1;-2,2];
rawLower = zeros(2,BI.l_dim);
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations = 123;
llFunctionEvaluations = 456;
rngBefore = rng;
[corrected,entry] = applyIndividualPredictionCorrection( ...
    CMA,rawUpper,rawLower,11,BI,options,state);
assert(norm(corrected(1,:)-[0.95,-0.95]) < 1e-12);
assert(norm(corrected(2,:)-[-1.75,1.75]) < 1e-12);
assert(isequal(corrected-entry.correction_vectors,rawUpper));
assert(all(abs(entry.correction_lengths - ...
    sqrt(sum(entry.correction_vectors.^2,2))) < 1e-12));
assert(all(entry.correction_applied));
assert(entry.prediction_valid_used && entry.prediction_source_generation == 10);
assert(isequal(entry.joint_after_correction,[corrected,rawLower]));
assert(isequal(rngBefore,rng));
assert(ulFunctionEvaluations == 123 && llFunctionEvaluations == 456);

% A second midpoint boundary repair follows the original sampler's rule.
boundedBI = BI;
boundedBI.u_lb = [-1,-1];
boundedBI.u_ub = [1,1];
state.pending.x_pred = [2,0];
[bounded,boundEntry] = applyIndividualPredictionCorrection( ...
    CMA,[0.9,0],zeros(1,boundedBI.l_dim),11,boundedBI,options,state);
assert(norm(bounded-[0.95,0]) < 1e-12);
assert(boundEntry.boundary_repaired && ...
       boundEntry.boundary_coordinate_repaired(1,1));

% Matched SMD1 test: population 10 produces x_pred, which can first affect
% population 11.  The normal CMA mean itself is never externally translated.
baseBI = getBLOPinfo('SMD',1,5);
baseBI.UmaxFEs = 400;
baseBI.UmaxImprFEs = 350;
baseBI.LmaxFEs = 250;
baseBI.LmaxImprFEs = 25;
baseBI.u_N = 50;
baseBI.l_N = 50;
baseBI.u_ftol = 0; % avoid a target stop before the timing test is complete
baseBI.generationDiagnostic = struct('enabled',true);
baseBI.enablePredictionCorrection = false;
baseBI.enableIndividualPredictionCorrection = false;
baseBI.individualPredictionCorrection = struct('logEnabled',true,'weight',0.1, ...
    'startGeneration',10);
seed = 1;
rng(seed,'twister');
baseline = BLCMAES(baseBI);

individualBI = baseBI;
individualBI.enableIndividualPredictionCorrection = true;
rng(seed,'twister');
individual = BLCMAES(individualBI);
log = individual.individualPredictionLog;
baselineLog = baseline.individualPredictionLog;
assert(all(~[log(1:min(10,end)).prediction_valid_used]), ...
    'A prediction was incorrectly used before generation 11.');
firstApplied = find(arrayfun(@(item)any(item.correction_applied),log),1);
assert(~isempty(firstApplied), ...
    'SMD1 test did not produce an applicable quadratic prediction.');
assert(firstApplied >= 11,'Generation 10 prediction must affect generation 11.');
entry = log(firstApplied);
previous = log(firstApplied-1);
assert(entry.prediction_valid_used && ...
       entry.prediction_source_generation == firstApplied-1);
assert(norm(entry.x_pred_used-previous.next_x_pred) < 1e-10);
assert(norm(entry.upper_after-entry.upper_before-entry.correction_vectors,'fro') < 1e-10);
assert(all(abs(entry.correction_lengths - ...
    sqrt(sum(entry.correction_vectors.^2,2))) < 1e-10));
assert(~entry.mean_directly_corrected);
assert(isequaln([log(1:10).cma_mean_after_update], ...
                [baselineLog(1:10).cma_mean_after_update]), ...
    'The mean differs before any v2 population correction can occur.');
assert(~isequaln(baseline.generationPopulationLog, ...
                 individual.generationPopulationLog), ...
    'Corrected offspring did not affect the subsequently evaluated population.');

report = struct('passed',true,'seed',seed, ...
    'firstAppliedGeneration',firstApplied, ...
    'correctionCount',sum(arrayfun(@(item)sum(item.correction_applied),log)), ...
    'baselineUF',baseline.UF,'individualUF',individual.UF, ...
    'baselineUFE',baseline.UFEs,'individualUFE',individual.UFEs, ...
    'baselineLFE',baseline.LFEs,'individualLFE',individual.LFEs);
save(fullfile(outputDir,'SMD1_individual_prediction_smoke.mat'), ...
    'baseline','individual','baseBI','individualBI','report','entry');
save(fullfile(outputDir,'test_report.mat'),'report');
disp(report);
end
