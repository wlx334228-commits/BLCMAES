function experiment = runSMDHighDimPredictionInjection(outputRoot,numberOfRuns,problemNumbers)
%RUNSMDHIGHDIMPREDICTIONINJECTION Formal (10+10) SMD experiment only.
% For every SMD1--SMD12 problem it runs two paired modes:
%   K0 = 12 normal CMA-ES candidates;
%   K6 = 6 normal + 1 direct prediction + 5 auxiliary samples.
% The former (5+5) experiment is deliberately not run by this script.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    % A fixed path lets an interrupted 20-run experiment resume safely.
    outputRoot = fullfile(projectRoot,'results_prediction_injection_highdim', ...
        'DU10_DL10_half_prediction_guided_20runs');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:12; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','>=',1});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',12});

smokeDir = fullfile(outputRoot,'smoke_test_DU10_DL10');
if exist(fullfile(smokeDir,'highdim_prediction_injection_test_report.mat'),'file') ~= 2
    if ~exist(outputRoot,'dir'), mkdir(outputRoot); end
    testHighDimPredictionInjection(smokeDir,20);
else
    fprintf('Reuse completed high-dimensional smoke test: %s\n',smokeDir);
end

dimensionSpecs = struct('totalDim',20,'upperDim',10,'lowerDim',10, ...
    'injectCount',6);
experiment = struct();
for specIndex = 1:numel(dimensionSpecs)
    spec = dimensionSpecs(specIndex);
    dimensionLabel = sprintf('DU%d_DL%d',spec.upperDim,spec.lowerDim);
    dimensionRoot = fullfile(outputRoot,dimensionLabel);
    [summary,pairedTests,runs] = runSMDPredictionInjection( ...
        dimensionRoot,numberOfRuns,problemNumbers,[0,spec.injectCount], ...
        'direct_plus_auxiliary',spec.totalDim);
    [fitRuns,fitSummary] = summarisePredictionInjectionFitStatistics(dimensionRoot);
    % Assign nested structures one field at a time.  Passing struct arrays
    % directly to struct(...) can create a dissimilar structure array when
    % their fields have dimension-dependent widths.
    caseReport = struct();
    caseReport.totalDim = spec.totalDim;
    caseReport.upperDim = spec.upperDim;
    caseReport.lowerDim = spec.lowerDim;
    caseReport.populationSize = 2*spec.injectCount;
    caseReport.predictionInjectCount = spec.injectCount;
    caseReport.normalCandidateCount = spec.injectCount;
    caseReport.directPredictionCount = 1;
    caseReport.auxiliaryCandidateCount = spec.injectCount-1;
    caseReport.outputRoot = dimensionRoot;
    caseReport.summary = summary;
    caseReport.pairedTests = pairedTests;
    caseReport.runs = runs;
    caseReport.fitRuns = fitRuns;
    caseReport.fitSummary = fitSummary;
    experiment.(dimensionLabel) = caseReport;
end
save(fullfile(outputRoot,'highdim_experiment_index.mat'),'experiment', ...
    'numberOfRuns','problemNumbers','dimensionSpecs');
fprintf('\nHigh-dimensional prediction-injection experiment completed.\n');
fprintf('  (10+10): %s\n',experiment.DU10_DL10.outputRoot);
end
