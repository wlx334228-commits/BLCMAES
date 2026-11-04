function experiment = runSMDAuxiliarySamplingScaleSensitivity( ...
    outputRoot,numberOfRuns,problemNumbers)
%RUNSMDAUXILIARYSAMPLINGSCALESENSITIVITY Formal eta study at (DU,DL)=(2,3).
% A0: 8 normal; A1: 7 normal plus a direct prediction; E025/E050/E075/E100:
% 5 normal + 1 direct prediction + 2 CMA-covariance Gaussian auxiliary
% samples, with the auxiliary upper perturbation scaled by eta.
%
% Run all SMD1--SMD8, 20 paired seeds, with:
%   runSMDAuxiliarySamplingScaleSensitivity

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_auxiliary_sampling_scale_sensitivity_2D','SMD1_8_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

specifications = scaleSpecifications(outputRoot);
saveOrValidateConfig(outputRoot,numberOfRuns,problemNumbers,specifications);
smokeRoot = fullfile(outputRoot,'smoke_test_scale');
if exist(fullfile(smokeRoot,'auxiliary_sampling_scale_test_report.mat'),'file') ~= 2
    testAuxiliarySamplingScaleSensitivity(smokeRoot);
else
    fprintf('Reuse completed auxiliary-scale smoke test: %s\n',smokeRoot);
end

for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    fprintf(['Run %s: %d normal + %d direct + %d auxiliary; ' ...
        'eta=%.2f.\n'],spec.id,spec.normalCount,spec.directCount, ...
        spec.auxiliaryCount,spec.auxiliarySamplingScale);
    runSMDPredictionInjection(spec.resultRoot,numberOfRuns,problemNumbers, ...
        spec.injectCount,'direct_plus_auxiliary',5,40,'cma_gaussian', ...
        spec.auxiliarySamplingScale);
end

runs = collectScaleRuns(specifications,problemNumbers,numberOfRuns);
[summary,pairedTests] = summariseAuxiliarySamplingScaleSensitivity( ...
    runs,specifications,problemNumbers);
writetable(runs,fullfile(outputRoot,'auxiliary_scale_runs.csv'));
writetable(summary,fullfile(outputRoot,'auxiliary_scale_summary.csv'));
writetable(pairedTests, ...
    fullfile(outputRoot,'auxiliary_scale_paired_wilcoxon.csv'));
experiment = struct('outputRoot',outputRoot,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'runs',runs,'summary',summary,'pairedTests',pairedTests);
save(fullfile(outputRoot,'auxiliary_scale_experiment_index.mat'), ...
    'experiment','-v7');
fprintf('\nAuxiliary-sampling eta sensitivity experiment completed: %s\n', ...
    outputRoot);
end

function specifications = scaleSpecifications(outputRoot)
specifications = [ ...
    makeSpec('A0',0,8,0,0,1.00,outputRoot); ...
    makeSpec('A1',1,7,1,0,1.00,outputRoot); ...
    makeSpec('E025',3,5,1,2,0.25,outputRoot); ...
    makeSpec('E050',3,5,1,2,0.50,outputRoot); ...
    makeSpec('E075',3,5,1,2,0.75,outputRoot); ...
    makeSpec('E100',3,5,1,2,1.00,outputRoot)];
end

function spec = makeSpec(id,injectCount,normalCount,directCount, ...
    auxiliaryCount,eta,outputRoot)
assert(normalCount+directCount+auxiliaryCount == 8, ...
    'Every (2+3) eta-study population must contain exactly eight candidates.');
spec = struct('id',id,'injectCount',injectCount, ...
    'normalCount',normalCount,'directCount',directCount, ...
    'auxiliaryCount',auxiliaryCount,'auxiliarySamplingScale',eta, ...
    'resultRoot',fullfile(outputRoot,sprintf('%s_eta_%03d',id,round(100*eta))));
end

function saveOrValidateConfig(outputRoot,numberOfRuns,problemNumbers,specifications)
configFile = fullfile(outputRoot,'auxiliary_scale_experiment_config.mat');
config = struct('totalDim',5,'upperDim',2,'lowerDim',3, ...
    'populationSize',8,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'comparisons',{{'A1_vs_A0','E025_vs_A0','E050_vs_A0','E075_vs_A0', ...
        'E100_vs_A0','E025_vs_A1','E050_vs_A1','E075_vs_A1', ...
        'E100_vs_A1','E025_vs_E100','E050_vs_E100','E075_vs_E100'}}, ...
    'notes',['Only the upper auxiliary perturbation scale eta differs ' ...
        'among E025/E050/E075/E100; eta=1 preserves current A2 sampling.']);
if exist(configFile,'file') ~= 2
    save(configFile,'config');
    return;
end
loaded = load(configFile,'config');
previous = loaded.config;
assert(previous.totalDim == 5 && previous.numberOfRuns == numberOfRuns && ...
    isequal(previous.problemNumbers,problemNumbers) && ...
    sameSpecificationDefinitions(previous.specifications,specifications), ...
    ['This output directory has a different eta-study configuration. ' ...
     'Choose a fresh outputRoot rather than mixing experiments.']);
end

function same = sameSpecificationDefinitions(previous,current)
% Result paths are intentionally excluded: the experiment directory can be
% copied to another computer and resumed there without changing its actual
% eta/population definition.
if numel(previous) ~= numel(current)
    same = false;
    return;
end
if isfield(previous,'resultRoot'), previous = rmfield(previous,'resultRoot'); end
if isfield(current,'resultRoot'), current = rmfield(current,'resultRoot'); end
same = isequal(previous,current);
end

function allRuns = collectScaleRuns(specifications,problemNumbers,numberOfRuns)
allRuns = table();
for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    fileName = fullfile(spec.resultRoot,'injection_runs.csv');
    assert(exist(fileName,'file') == 2,'Missing result table: %s',fileName);
    rows = readtable(fileName);
    rows = rows(ismember(rows.SMD,problemNumbers) & ...
        rows.prediction_inject_count == spec.injectCount,:);
    validateSchedule(rows,spec,problemNumbers,numberOfRuns);
    rows.experiment_id = repmat({spec.id},height(rows),1);
    rows.auxiliary_sampling_scale = repmat( ...
        spec.auxiliarySamplingScale,height(rows),1);
    rows.configured_normal_candidate_count = repmat(spec.normalCount,height(rows),1);
    rows.configured_direct_candidate_count = repmat(spec.directCount,height(rows),1);
    rows.configured_auxiliary_candidate_count = repmat( ...
        spec.auxiliaryCount,height(rows),1);
    rows.result_root = repmat({spec.resultRoot},height(rows),1);
    if isempty(allRuns), allRuns = rows; else, allRuns = [allRuns;rows]; end %#ok<AGROW>
end
allRuns = sortrows(allRuns,{'SMD','experiment_id','run_no'});
end

function validateSchedule(rows,spec,problemNumbers,numberOfRuns)
for smdNo = problemNumbers(:)'
    subset = rows(rows.SMD == smdNo,:);
    assert(height(subset) == numberOfRuns, ...
        '%s has an incomplete SMD%d run set.',spec.id,smdNo);
    assert(isequal(sort(subset.run_no)',1:numberOfRuns), ...
        '%s has duplicate/missing runs for SMD%d.',spec.id,smdNo);
    expectedSeed = (smdNo-1)*numberOfRuns + subset.run_no;
    assert(all(subset.seed == expectedSeed), ...
        '%s does not use the paired seed schedule for SMD%d.',spec.id,smdNo);
end
end
