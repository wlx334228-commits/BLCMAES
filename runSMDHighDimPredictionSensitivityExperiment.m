function experiment = runSMDHighDimPredictionSensitivityExperiment( ...
    outputRoot,numberOfRuns,problemNumbers,referenceRoot)
%RUNSMDHIGHDIMPREDICTIONSENSITIVITY 10-D M/K sensitivity experiment.
% This is the single entry point for the requested (10+10) study.
% When the previous paired A0/E0 records are present, it reuses them and
% runs only:
%   E1: M=300, K=3 = 9 normal + 1 direct + 2 auxiliary;
%   E2: M=250, K=3 = 9 normal + 1 direct + 2 auxiliary;
%   E3: M=200, K=3 = 9 normal + 1 direct + 2 auxiliary;
%   E4: M=100, K=3 = 9 normal + 1 direct + 2 auxiliary.
% If the previous result folder is absent (for example after copying the
% project to a different computer), it automatically runs A0 and E0 too.
% All modes use the same 20 seeds.  The normal CMA-ES and lower-level
% optimizer are untouched; M only changes the existing archive-fit option.
%
% Example (formal 12-problem, 20-run study):
%   runSMDHighDimPredictionSensitivityExperiment
%
% Example (resumable subset for a quick check):
%   runSMDHighDimPredictionSensitivityExperiment([],20,[1 2])

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_prediction_injection_sensitivity_10D', ...
        'DU10_DL10_E0_E4_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:12; end
if nargin < 4 || isempty(referenceRoot)
    referenceRoot = fullfile(projectRoot, ...
        'results_prediction_injection_highdim(10+10)', ...
        'DU10_DL10_half_prediction_guided_20runs','DU10_DL10');
end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',12});
assert(numberOfRuns == 20, ...
    ['This study reuses the completed A0/E0 records, whose paired seed ' ...
     'schedule is fixed at 20 runs.']);
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');
referenceAvailable = exist(fullfile(referenceRoot,'injection_runs.csv'), ...
    'file') == 2;
if referenceAvailable
    fprintf('Reuse existing paired A0/E0 results from: %s\n',referenceRoot);
else
    warning('BLCMAES:MissingReusableReference', ...
        ['Previous A0/E0 results were not found at:\n  %s\n' ...
         'This computer will run A0 and E0 from scratch, alongside E1--E4.'], ...
        referenceRoot);
end

if ~exist(outputRoot,'dir'), mkdir(outputRoot); end
specifications = makeSpecifications(outputRoot,referenceRoot,referenceAvailable);
saveOrValidateConfiguration(outputRoot,numberOfRuns,problemNumbers, ...
    referenceRoot,specifications);

% This validates complete 66-coefficient fitting and the E0/E1 candidate
% compositions without calling any SMD objective or lower-level solver.
smokeRoot = fullfile(outputRoot,'smoke_test_DU10_DL10');
if exist(fullfile(smokeRoot,'highdim_prediction_sensitivity_test_report.mat'), ...
        'file') ~= 2
    testHighDimPredictionSensitivity(smokeRoot,[100 200 250 300],[3 6]);
else
    fprintf('Reuse completed sensitivity smoke test: %s\n',smokeRoot);
end

for i = 1:numel(specifications)
    spec = specifications(i);
    if spec.reuseCompleted
        validateReusableReference(spec.resultRoot,spec,problemNumbers,numberOfRuns);
        fprintf('Reuse %s from %s\n',spec.id,spec.resultRoot);
    else
        fprintf(['Run %s: M=%d, %d normal + %d direct + %d auxiliary ' ...
            'candidates.\n'],spec.id,spec.sampleCount,spec.normalCount, ...
            spec.directCount,spec.auxiliaryCount);
        runSMDPredictionInjection(spec.resultRoot,numberOfRuns, ...
            problemNumbers,spec.injectCount,'direct_plus_auxiliary',20, ...
            spec.sampleCount);
    end
end

[runs,fitRuns] = collectResults(specifications,problemNumbers,numberOfRuns);
[summary,pairedTests] = summariseHighDimPredictionSensitivity( ...
    runs,fitRuns,specifications,problemNumbers);
writetable(runs,fullfile(outputRoot,'sensitivity_runs.csv'));
writetable(fitRuns,fullfile(outputRoot,'sensitivity_fit_runs.csv'));
writetable(summary,fullfile(outputRoot,'sensitivity_summary.csv'));
writetable(pairedTests,fullfile(outputRoot,'sensitivity_paired_wilcoxon.csv'));
experiment = struct();
experiment.outputRoot = outputRoot;
experiment.referenceRoot = referenceRoot;
experiment.numberOfRuns = numberOfRuns;
experiment.problemNumbers = problemNumbers;
experiment.specifications = specifications;
experiment.runs = runs;
experiment.fitRuns = fitRuns;
experiment.summary = summary;
experiment.pairedTests = pairedTests;
save(fullfile(outputRoot,'sensitivity_experiment_index.mat'),'experiment', ...
    '-v7');

fprintf('\n10-D sensitivity experiment complete: %s\n',outputRoot);
if specifications(1).reuseCompleted && specifications(2).reuseCompleted
    fprintf('A0/E0 were reused; E1--E4 have independent resumable directories.\n');
else
    fprintf('A0, E0, and E1--E4 were all generated in independent resumable directories.\n');
end
end

function specifications = makeSpecifications(outputRoot,referenceRoot,referenceAvailable)
% One direct candidate is included whenever K>0.  K=3 therefore means
% exactly 9 normal + 1 direct prediction + 2 auxiliary samples.
specifications = repmat(struct('id','','sampleCount',NaN,'injectCount',NaN, ...
    'normalCount',NaN,'directCount',NaN,'auxiliaryCount',NaN, ...
    'resultRoot','','reuseCompleted',false),1,6);
if referenceAvailable
    a0Root = referenceRoot;
    e0Root = referenceRoot;
else
    % Keep the newly generated baseline and E0 in separate resumable
    % directories.  This avoids mixing their K-specific experiment configs.
    a0Root = fullfile(outputRoot,'A0_M300_N12');
    e0Root = fullfile(outputRoot,'E0_M300_N6_D1_A5');
end
specifications(1) = makeSpec('A0',300,0,12,0,0,a0Root,referenceAvailable);
specifications(2) = makeSpec('E0',300,6,6,1,5,e0Root,referenceAvailable);
specifications(3) = makeSpec('E1',300,3,9,1,2, ...
    fullfile(outputRoot,'E1_M300_N9_D1_A2'),false);
specifications(4) = makeSpec('E2',250,3,9,1,2, ...
    fullfile(outputRoot,'E2_M250_N9_D1_A2'),false);
specifications(5) = makeSpec('E3',200,3,9,1,2, ...
    fullfile(outputRoot,'E3_M200_N9_D1_A2'),false);
specifications(6) = makeSpec('E4',100,3,9,1,2, ...
    fullfile(outputRoot,'E4_M100_N9_D1_A2'),false);
end

function spec = makeSpec(id,sampleCount,injectCount,normalCount,directCount, ...
    auxiliaryCount,resultRoot,reuseCompleted)
assert(normalCount + directCount + auxiliaryCount == 12, ...
    'Every 10-D candidate layout must contain exactly 12 individuals.');
spec = struct('id',id,'sampleCount',sampleCount,'injectCount',injectCount, ...
    'normalCount',normalCount,'directCount',directCount, ...
    'auxiliaryCount',auxiliaryCount,'resultRoot',resultRoot, ...
    'reuseCompleted',reuseCompleted);
end

function saveOrValidateConfiguration(outputRoot,numberOfRuns,problemNumbers, ...
    referenceRoot,specifications)
configFile = fullfile(outputRoot,'sensitivity_experiment_config.mat');
config = struct('totalDim',20,'upperDim',10,'lowerDim',10, ...
    'populationSize',12,'quadraticCoefficientCount',66, ...
    'numberOfRuns',numberOfRuns,'problemNumbers',problemNumbers, ...
    'referenceRoot',referenceRoot,'specifications',specifications, ...
    'notes',['M changes both local sample size and the earliest possible ' ...
        'prediction generation under the existing sample-availability rule.']);
if exist(configFile,'file') ~= 2
    save(configFile,'config');
    return;
end
loaded = load(configFile,'config');
previous = loaded.config;
sameSpecifications = isfield(previous,'specifications') && ...
    isequal(previous.specifications,config.specifications);
assert(previous.totalDim == config.totalDim && ...
    previous.numberOfRuns == config.numberOfRuns && ...
    isequal(previous.problemNumbers,config.problemNumbers) && ...
    strcmp(previous.referenceRoot,config.referenceRoot) && ...
    sameSpecifications, ...
    ['This output directory has a different sensitivity configuration. ' ...
     'Choose a new outputRoot instead of mixing experiments.']);
end

function validateReusableReference(referenceRoot,spec,problemNumbers,numberOfRuns)
runs = readtable(fullfile(referenceRoot,'injection_runs.csv'));
required = {'SMD','prediction_inject_count','run_no','seed'};
assert(all(ismember(required,runs.Properties.VariableNames)), ...
    'The reusable result table lacks required paired-run columns.');
for smdNo = problemNumbers(:)'
    rows = runs(runs.SMD == smdNo & ...
        runs.prediction_inject_count == spec.injectCount,:);
    assert(height(rows) == numberOfRuns, ...
        'Reference %s lacks %d completed runs for SMD%d.', ...
        spec.id,numberOfRuns,smdNo);
    assert(isequal(sort(rows.run_no)',1:numberOfRuns), ...
        'Reference %s has an incomplete run-number schedule for SMD%d.', ...
        spec.id,smdNo);
    expectedSeeds = (smdNo-1)*numberOfRuns + rows.run_no;
    assert(all(rows.seed == expectedSeeds), ...
        'Reference %s uses an incompatible paired seed schedule for SMD%d.', ...
        spec.id,smdNo);
end
end

function [allRuns,allFitRuns] = collectResults(specifications,problemNumbers, ...
    numberOfRuns)
allRuns = table();
allFitRuns = table();
processedRoots = cell(0,1);
rootFitRuns = cell(0,1);
for i = 1:numel(specifications)
    spec = specifications(i);
    runFile = fullfile(spec.resultRoot,'injection_runs.csv');
    assert(exist(runFile,'file') == 2,'Missing result table: %s',runFile);
    rawRuns = readtable(runFile);
    rows = rawRuns(ismember(rawRuns.SMD,problemNumbers) & ...
        rawRuns.prediction_inject_count == spec.injectCount,:);
    validateRunSchedule(rows,spec,problemNumbers,numberOfRuns);
    rows = attachSpecColumns(rows,spec);
    if isempty(allRuns), allRuns = rows; else, allRuns = [allRuns;rows]; end %#ok<AGROW>

    rootIndex = find(strcmp(processedRoots,spec.resultRoot),1);
    if isempty(rootIndex)
        [rootFit,~] = summarisePredictionInjectionFitStatistics( ...
            spec.resultRoot,false);
        processedRoots{end+1,1} = spec.resultRoot; %#ok<AGROW>
        rootFitRuns{end+1,1} = rootFit; %#ok<AGROW>
        rootIndex = numel(processedRoots);
    end
    rawFit = rootFitRuns{rootIndex};
    fitRows = rawFit(ismember(rawFit.SMD,problemNumbers) & ...
        rawFit.prediction_inject_count == spec.injectCount,:);
    assert(height(fitRows) == height(rows), ...
        'Fit-log rows do not match completed run rows for %s.',spec.id);
    fitRows = attachSpecColumns(fitRows,spec);
    if isempty(allFitRuns), allFitRuns = fitRows; else, allFitRuns = [allFitRuns;fitRows]; end %#ok<AGROW>
end
allRuns = sortrows(allRuns,{'SMD','experiment_id','run_no'});
allFitRuns = sortrows(allFitRuns,{'SMD','experiment_id','run_no'});
end

function validateRunSchedule(rows,spec,problemNumbers,numberOfRuns)
for smdNo = problemNumbers(:)'
    subset = rows(rows.SMD == smdNo,:);
    assert(height(subset) == numberOfRuns, ...
        '%s has %d rather than %d runs for SMD%d.', ...
        spec.id,height(subset),numberOfRuns,smdNo);
    assert(isequal(sort(subset.run_no)',1:numberOfRuns), ...
        '%s has missing or duplicate run numbers for SMD%d.',spec.id,smdNo);
    expectedSeeds = (smdNo-1)*numberOfRuns + subset.run_no;
    assert(all(subset.seed == expectedSeeds), ...
        '%s does not use the required paired seeds for SMD%d.',spec.id,smdNo);
end
end

function rows = attachSpecColumns(rows,spec)
rowCount = height(rows);
rows.experiment_id = repmat({spec.id},rowCount,1);
rows.archive_sample_count = repmat(spec.sampleCount,rowCount,1);
rows.normal_candidate_count = repmat(spec.normalCount,rowCount,1);
rows.direct_candidate_count_configured = repmat(spec.directCount,rowCount,1);
rows.auxiliary_candidate_count_configured = repmat(spec.auxiliaryCount,rowCount,1);
rows.result_root = repmat({spec.resultRoot},rowCount,1);
end
