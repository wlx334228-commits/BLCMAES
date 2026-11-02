function experiment = runSMDAuxiliarySamplingAblation( ...
    outputRoot,numberOfRuns,problemNumbers)
%RUNSMDAUXILIARYSAMPLINGABLATION Formal A0--A4 auxiliary-distribution study.
% Scope: unconstrained SMD1--SMD8 at (DU,DL)=(2,3), lambda=8, with 20
% paired seeds by default.  It compares only the candidate composition and
% the upper auxiliary distribution; the quadratic model and CMA-ES core are
% supplied unchanged by runSMDPredictionInjection.
%
%   A0: 8 normal candidates, no model;
%   A1: 7 normal + 1 direct prediction;
%   A2: 5 normal + 1 direct + 2 CMA-covariance Gaussian auxiliaries;
%   A3: 5 normal + 1 direct + 2 isotropic Gaussian auxiliaries;
%   A4: 5 normal + 1 direct + 2 uniform-ellipsoid auxiliaries.
%
% Run the formal experiment with:
%   runSMDAuxiliarySamplingAblation

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_auxiliary_sampling_ablation_2D', ...
        'SMD1_8_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:8; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'problemNumbers must not contain duplicates.');

if ~exist(outputRoot,'dir'), mkdir(outputRoot); end
specifications = ablationSpecifications(outputRoot);
saveOrValidateConfig(outputRoot,numberOfRuns,problemNumbers,specifications);

smokeRoot = fullfile(outputRoot,'smoke_test_A0_A4');
if exist(fullfile(smokeRoot,'auxiliary_sampling_ablation_test_report.mat'), ...
        'file') ~= 2
    testAuxiliarySamplingAblation(smokeRoot);
else
    fprintf('Reuse completed A0--A4 sampling smoke test: %s\n',smokeRoot);
end

for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    fprintf(['Run %s: %d normal + %d direct + %d auxiliary; ' ...
        'distribution=%s.\n'],spec.id,spec.normalCount,spec.directCount, ...
        spec.auxiliaryCount,spec.auxiliarySamplingDistribution);
    runSMDPredictionInjection(spec.resultRoot,numberOfRuns,problemNumbers, ...
        spec.injectCount,'direct_plus_auxiliary',5,40, ...
        spec.auxiliarySamplingDistribution);
end

runs = collectAblationRuns(specifications,problemNumbers,numberOfRuns);
[summary,pairedTests] = summariseAuxiliarySamplingAblation( ...
    runs,specifications,problemNumbers);
writetable(runs,fullfile(outputRoot,'auxiliary_sampling_runs.csv'));
writetable(summary,fullfile(outputRoot,'auxiliary_sampling_summary.csv'));
writetable(pairedTests, ...
    fullfile(outputRoot,'auxiliary_sampling_paired_wilcoxon.csv'));
experiment = struct('outputRoot',outputRoot,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'runs',runs,'summary',summary,'pairedTests',pairedTests);
save(fullfile(outputRoot,'auxiliary_sampling_experiment_index.mat'), ...
    'experiment','-v7');
fprintf('\nA0--A4 auxiliary-sampling ablation completed: %s\n',outputRoot);
end

function specifications = ablationSpecifications(outputRoot)
specifications = [ ...
    makeSpec('A0',0,8,0,0,'cma_gaussian',outputRoot); ...
    makeSpec('A1',1,7,1,0,'cma_gaussian',outputRoot); ...
    makeSpec('A2',3,5,1,2,'cma_gaussian',outputRoot); ...
    makeSpec('A3',3,5,1,2,'isotropic_gaussian',outputRoot); ...
    makeSpec('A4',3,5,1,2,'uniform_ellipsoid',outputRoot)];
end

function spec = makeSpec(id,injectCount,normalCount,directCount, ...
    auxiliaryCount,distribution,outputRoot)
assert(normalCount+directCount+auxiliaryCount == 8, ...
    'Each (2+3) ablation population must contain exactly eight candidates.');
spec = struct('id',id,'injectCount',injectCount, ...
    'normalCount',normalCount,'directCount',directCount, ...
    'auxiliaryCount',auxiliaryCount, ...
    'auxiliarySamplingDistribution',distribution, ...
    'resultRoot',fullfile(outputRoot,[id '_' distribution]));
end

function saveOrValidateConfig(outputRoot,numberOfRuns,problemNumbers,specifications)
configFile = fullfile(outputRoot,'auxiliary_sampling_experiment_config.mat');
config = struct('totalDim',5,'upperDim',2,'lowerDim',3, ...
    'populationSize',8,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers,'specifications',specifications, ...
    'comparisons',{{'A1_vs_A0','A2_vs_A1','A3_vs_A2','A4_vs_A2'}}, ...
    'notes',['A2/A3/A4 use the same K=3 and x_pred.  Only the upper ' ...
        'auxiliary distribution changes; lower coordinates use the ' ...
        'original conditional CMA-ES Gaussian relationship.']);
if exist(configFile,'file') ~= 2
    save(configFile,'config');
    return;
end
loaded = load(configFile,'config');
previous = loaded.config;
assert(previous.totalDim == 5 && previous.numberOfRuns == numberOfRuns && ...
    isequal(previous.problemNumbers,problemNumbers) && ...
    isequal(previous.specifications,specifications), ...
    ['This output directory has a different A0--A4 configuration. ' ...
     'Choose a fresh outputRoot rather than mixing experiments.']);
end

function allRuns = collectAblationRuns(specifications,problemNumbers,numberOfRuns)
allRuns = table();
for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    fileName = fullfile(spec.resultRoot,'injection_runs.csv');
    assert(exist(fileName,'file') == 2,'Missing result table: %s',fileName);
    rawRuns = readtable(fileName);
    rows = rawRuns(ismember(rawRuns.SMD,problemNumbers) & ...
        rawRuns.prediction_inject_count == spec.injectCount,:);
    validateSchedule(rows,spec,problemNumbers,numberOfRuns);
    rows.experiment_id = repmat({spec.id},height(rows),1);
    rows.auxiliary_sampling_distribution = repmat( ...
        {spec.auxiliarySamplingDistribution},height(rows),1);
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
