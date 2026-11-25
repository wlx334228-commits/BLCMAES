function experiment = runSMDFixedAuxiliaryScaleComparison( ...
    outputRoot,numberOfRuns,problemNumbers,studyOptions)
%RUNSMDFIXEDAUXILIARYSCALECOMPARISON Complete A0/A1/K3-scale comparison.
% At (DU,DL)=(2,3), every listed variant runs independently from its
% initial population to the original BL-CMA-ES termination condition.  The
% reported upper/lower errors and UFE/LFE are therefore terminal metrics,
% not a one-generation diagnostic quantity.
%
% With no studyOptions, retain the historical four-arm study:
%   A0, A1, K3_eta025, K3_eta050.
% A caller can select an arbitrary K3 auxiliary scale list and choose
% whether to include A1, for example:
%   runSMDFixedAuxiliaryScaleComparison([],10,1:8, ...
%       struct('etas',[.10 .15 .20 .25],'includeA1',false));
% Set includeA2=true for 6 normal + 2 auxiliary-only candidates. Its
% independent scale is auxiliaryOnlyEta (default .20).
% Positive etas above one enlarge the auxiliary upper perturbations.
% comparisonReferenceEta selects the K3 reference (default: largest eta).
% significanceMode='raw' uses unadjusted p<.05 for per-metric advantages;
% the legacy default 'holm' retains its correction and success safeguards.
%
% The default is the requested formal study: SMD1--SMD8, 20 paired seeds.
% It is resumable: each leaf experiment writes its own CSV and MAT records
% after every completed run.  Results from other eta studies are never read.
%
%   runSMDFixedAuxiliaryScaleComparison
%   runSMDFixedAuxiliaryScaleComparison([],1,1)  % small smoke batch

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_fixed_auxiliary_scale_comparison_2D','SMD1_8_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
if nargin < 4 || isempty(studyOptions), studyOptions = struct(); end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

studyOptions = normaliseStudyOptions(studyOptions);
specifications = fixedScaleSpecifications(outputRoot,studyOptions);
comparisons = fixedScaleComparisons(studyOptions);
saveOrValidateConfiguration(outputRoot,numberOfRuns,problemNumbers, ...
    specifications,comparisons,studyOptions.significanceMode);
snapshotFormalSources(projectRoot,outputRoot);

% This is a sampling-only validation; it performs no upper/lower objective
% evaluations.  The formal optimizer runs begin only after it succeeds.
smokeRoot = fullfile(outputRoot,'smoke_test_sampling');
if exist(fullfile(smokeRoot,'auxiliary_sampling_scale_test_report.mat'), ...
        'file') ~= 2
    samplingEtas = studyOptions.etas;
    if studyOptions.includeA2
        samplingEtas = unique([samplingEtas,studyOptions.auxiliaryOnlyEta]);
    end
    testAuxiliarySamplingScaleSensitivity(smokeRoot,2000,samplingEtas);
else
    fprintf('Reuse completed sampling smoke test: %s\n',smokeRoot);
end

for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    fprintf(['Run %s: %d normal + %d direct + %d auxiliary; ' ...
        'auxiliary eta=%s.\n'],spec.id,spec.normalCount,spec.directCount, ...
        spec.auxiliaryCount,displayEta(spec.reportedEta));
    runSMDPredictionInjection(spec.resultRoot,numberOfRuns,problemNumbers, ...
        spec.injectCount,spec.candidatePolicy,5,40,'cma_gaussian', ...
        spec.runnerEta);
end

runs = collectFixedScaleRuns(specifications,problemNumbers,numberOfRuns);
[summary,pairedTests] = summariseFixedAuxiliaryScaleComparison( ...
    runs,specifications,problemNumbers,comparisons,studyOptions.significanceMode);
[matchedBudgetRuns,matchedBudgetSummary,matchedBudgetTests] = ...
    analyseFixedAuxiliaryScaleMatchedBudget(outputRoot,runs, ...
    problemNumbers,comparisons,studyOptions.significanceMode);
writetable(runs,fullfile(outputRoot,'fixed_scale_runs.csv'));
writetable(summary,fullfile(outputRoot,'fixed_scale_summary.csv'));
writetable(pairedTests, ...
    fullfile(outputRoot,'fixed_scale_paired_wilcoxon.csv'));

experiment = struct('outputRoot',outputRoot,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'comparisons',{comparisons},'runs',runs,'summary',summary, ...
    'pairedTests',pairedTests,'matchedBudgetRuns',matchedBudgetRuns, ...
    'matchedBudgetSummary',matchedBudgetSummary, ...
    'matchedBudgetTests',matchedBudgetTests);
save(fullfile(outputRoot,'fixed_scale_experiment_index.mat'), ...
    'experiment','-v7');
fprintf(['\nFixed auxiliary-scale comparison completed: %s\n' ...
    'Files: fixed_scale_runs.csv, fixed_scale_summary.csv, ' ...
    'fixed_scale_paired_wilcoxon.csv\n'],outputRoot);
end

function options = normaliseStudyOptions(options)
assert(isstruct(options) && isscalar(options), ...
    'studyOptions must be a scalar structure.');
defaults = struct('etas',[.25 .50],'includeA1',true, ...
    'includeA2',false,'auxiliaryOnlyEta',.20, ...
    'comparisonReferenceEta',NaN,'significanceMode','holm');
names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(options,name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end
validateattributes(options.etas,{'numeric'}, ...
    {'real','finite','vector','nonempty','positive'});
options.etas = sort(unique(double(options.etas(:)')));
validateattributes(options.comparisonReferenceEta,{'numeric'}, ...
    {'real','scalar'});
if ~isnan(options.comparisonReferenceEta)
    assert(ismember(options.comparisonReferenceEta,options.etas), ...
        'comparisonReferenceEta must be one of the configured etas.');
end
options.significanceMode = validatestring(options.significanceMode,{'raw','holm'});
validateattributes(options.includeA1,{'logical','numeric'}, {'scalar'});
options.includeA1 = logical(options.includeA1);
validateattributes(options.includeA2,{'logical','numeric'}, {'scalar'});
options.includeA2 = logical(options.includeA2);
validateattributes(options.auxiliaryOnlyEta,{'numeric'}, ...
    {'real','finite','scalar','positive','<=',1});
end

function specifications = fixedScaleSpecifications(outputRoot,options)
specifications = makeSpec('A0',0,8,0,0,0.50,NaN,outputRoot);
if options.includeA1
    specifications(end+1,1) = makeSpec('A1',1,7,1,0,0.50,NaN,outputRoot); %#ok<AGROW>
end
if options.includeA2
    specifications(end+1,1) = makeSpec('A2',2,6,0,2, ...
        options.auxiliaryOnlyEta,options.auxiliaryOnlyEta,outputRoot); %#ok<AGROW>
end
for eta = options.etas
    specifications(end+1,1) = makeSpec(scaleVariantId(eta),3,5,1,2, ...
        eta,eta,outputRoot); %#ok<AGROW>
end
end

function spec = makeSpec(id,injectCount,normalCount,directCount, ...
    auxiliaryCount,runnerEta,reportedEta,outputRoot)
assert(normalCount + directCount + auxiliaryCount == 8, ...
    'Every (2+3) population must contain exactly eight candidates.');
spec = struct('id',id,'injectCount',injectCount, ...
    'normalCount',normalCount,'directCount',directCount, ...
    'auxiliaryCount',auxiliaryCount,'runnerEta',runnerEta, ...
    'reportedEta',reportedEta,'resultRoot',fullfile(outputRoot,id), ...
    'candidatePolicy','direct_plus_auxiliary');
if auxiliaryCount > 0 && directCount == 0
    spec.candidatePolicy = 'auxiliary_only';
end
end

function comparisons = fixedScaleComparisons(options)
% Every configured K3 scale is compared with A0 using paired seeds.  A1,
% if enabled, is also compared with A0 and each K3 variant.
comparisons = cell(0,3);
if options.includeA1
    comparisons(end+1,:) = {'A1_vs_A0','A0','A1'}; %#ok<AGROW>
end
if options.includeA2
    comparisons(end+1,:) = {'A2_vs_A0','A0','A2'}; %#ok<AGROW>
    if options.includeA1
        comparisons(end+1,:) = {'A2_vs_A1','A1','A2'}; %#ok<AGROW>
    end
end
for eta = options.etas
    id = scaleVariantId(eta);
    comparisons(end+1,:) = {[id '_vs_A0'],'A0',id}; %#ok<AGROW>
    if options.includeA1
        comparisons(end+1,:) = {[id '_vs_A1'],'A1',id}; %#ok<AGROW>
    end
    if options.includeA2
        comparisons(end+1,:) = {[id '_vs_A2'],'A2',id}; %#ok<AGROW>
    end
end
if numel(options.etas) >= 2
    referenceEta = options.comparisonReferenceEta;
    if isnan(referenceEta), referenceEta = options.etas(end); end
    referenceId = scaleVariantId(referenceEta);
    for eta = options.etas(options.etas ~= referenceEta)
        id = scaleVariantId(eta);
        comparisons(end+1,:) = {[id '_vs_' referenceId],referenceId,id}; %#ok<AGROW>
    end
end
end

function id = scaleVariantId(eta)
id = sprintf('K3_eta%03d',round(100*eta));
end

function saveOrValidateConfiguration(outputRoot,numberOfRuns,problemNumbers, ...
    specifications,comparisons,significanceMode)
configFile = fullfile(outputRoot,'fixed_scale_experiment_config.mat');
config = struct('totalDim',5,'upperDim',2,'lowerDim',3, ...
    'populationSize',8,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'sampleCount',40, ...
    'startGeneration',10,'historyGenerations',10, ...
    'candidatePolicy','per_variant_see_specifications', ...
    'auxiliarySamplingDistribution','cma_gaussian', ...
    'predictionModelType','quadratic', ...
    'seedRule','seed=(SMD-1)*numberOfRuns+run_no', ...
    'specifications',serialiseSpecifications(specifications), ...
    'comparisons',{comparisons},'significanceMode',significanceMode, ...
    'notes',['K3 variants differ only in auxiliary scale. A2 uses two ' ...
        'auxiliary-only candidates and no direct point; its scale and policy ' ...
        'are stored in specifications. A0/A1 contain no auxiliary samples.']);
if exist(configFile,'file') ~= 2
    save(configFile,'config');
    return;
end
loaded = load(configFile,'config');
previous = loaded.config;
if ~isfield(previous,'significanceMode'), previous.significanceMode = 'holm'; end
% Older four/five-arm configurations predate this explicit per-arm field.
if ~isfield(previous.specifications,'candidatePolicy')
    for i = 1:numel(previous.specifications)
        previous.specifications(i).candidatePolicy = 'direct_plus_auxiliary';
    end
end
assert(previous.totalDim == config.totalDim && ...
    previous.numberOfRuns == config.numberOfRuns && ...
    isequal(previous.problemNumbers,config.problemNumbers) && ...
    isequaln(previous.specifications,config.specifications) && ...
    strcmp(previous.significanceMode,config.significanceMode) && ...
    isequal(previous.comparisons,config.comparisons), ...
    ['This directory has a different fixed-scale experiment configuration. ' ...
     'Choose a fresh outputRoot rather than mixing results.']);
end

function stored = serialiseSpecifications(specifications)
stored = specifications;
for i = 1:numel(stored)
    stored(i).resultRoot = stored(i).id;
end
end

function allRuns = collectFixedScaleRuns(specifications,problemNumbers,numberOfRuns)
allRuns = table();
for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    runFile = fullfile(spec.resultRoot,'injection_runs.csv');
    assert(exist(runFile,'file') == 2,'Missing run table: %s',runFile);
    rows = readtable(runFile);
    rows = rows(ismember(rows.SMD,problemNumbers) & ...
        rows.prediction_inject_count == spec.injectCount,:);
    validatePairedSchedule(rows,spec,problemNumbers,numberOfRuns);
    rows.variant = repmat({spec.id},height(rows),1);
    rows.auxiliary_sampling_scale = repmat(spec.reportedEta,height(rows),1);
    rows.configured_normal_candidate_count = repmat(spec.normalCount,height(rows),1);
    rows.configured_direct_candidate_count = repmat(spec.directCount,height(rows),1);
    rows.configured_auxiliary_candidate_count = repmat( ...
        spec.auxiliaryCount,height(rows),1);
    rows.result_directory = repmat({spec.id},height(rows),1);
    if isempty(allRuns), allRuns = rows; else, allRuns = [allRuns;rows]; end %#ok<AGROW>
end
allRuns = sortrows(allRuns,{'SMD','variant','run_no'});
end

function validatePairedSchedule(rows,spec,problemNumbers,numberOfRuns)
for smdNo = problemNumbers(:)'
    subset = rows(rows.SMD == smdNo,:);
    assert(height(subset) == numberOfRuns, ...
        '%s has an incomplete SMD%d run set.',spec.id,smdNo);
    assert(isequal(sort(subset.run_no)',1:numberOfRuns), ...
        '%s has duplicate/missing runs for SMD%d.',spec.id,smdNo);
    expectedSeeds = (smdNo-1)*numberOfRuns + subset.run_no;
    assert(all(subset.seed == expectedSeeds), ...
        '%s does not use the configured paired seed schedule for SMD%d.', ...
        spec.id,smdNo);
    assert(all(subset.consistency_passed ~= 0), ...
        '%s has a failed consistency check in SMD%d.',spec.id,smdNo);
end
end

function snapshotFormalSources(projectRoot,outputRoot)
snapshotRoot = fullfile(outputRoot,'source_snapshot');
if exist(snapshotRoot,'dir') ~= 7, mkdir(snapshotRoot); end
files = {'BLCMAES.m','runSMDPredictionInjection.m', ...
    'runSMDFixedAuxiliaryScaleComparison.m', ...
    'runSMDSmallAuxiliaryScaleExperiment.m', ...
    'runSMDSmallAuxiliaryScaleFullComparison.m', ...
    'testSmallAuxiliaryScaleFullComparison.m', ...
    'runSMDLargeAuxiliaryScaleExperiment.m', ...
    'testLargeAuxiliaryScaleExperiment.m', ...
    'collectLargeAuxiliaryScaleDiagnostics.m', ...
    'summariseFixedAuxiliaryScaleComparison.m', ...
    'analyseFixedAuxiliaryScaleMatchedBudget.m', ...
    'testFixedAuxiliaryScaleComparison.m', ...
    'applyPredictionInjection.m','predictionInjectionOptions.m', ...
    'fitLocalUpperQuadratic.m','pairedWilcoxonSignedRank.m', ...
    'testAuxiliarySamplingScaleSensitivity.m'};
for i = 1:numel(files)
    sourceFile = fullfile(projectRoot,files{i});
    if exist(sourceFile,'file') == 2
        copyfile(sourceFile,snapshotRoot);
    end
end
end

function text = displayEta(eta)
if isfinite(eta), text = sprintf('%.2f',eta); else, text = 'not applicable'; end
end
