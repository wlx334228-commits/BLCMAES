function [fitRuns,fitSummary] = summarisePredictionInjectionFitStatistics( ...
    outputRoot,writeOutputs)
%SUMMARISEPREDICTIONINJECTIONFITSTATISTICS Summarise model-fit outcomes.
% This is offline bookkeeping only.  It reads each saved injection log and
% adds no upper/lower evaluation or random-number call to BL-CMA-ES.

if nargin < 2 || isempty(writeOutputs), writeOutputs = true; end
validateattributes(writeOutputs,{'logical','numeric'},{'scalar','binary'});
writeOutputs = logical(writeOutputs);

runFile = fullfile(outputRoot,'injection_runs.csv');
assert(exist(runFile,'file') == 2,'Missing run table: %s',runFile);
runs = readtable(runFile);
required = {'SMD','prediction_inject_count','run_no','mat_file'};
assert(all(ismember(required,runs.Properties.VariableNames)), ...
    'Run table does not contain prediction-injection metadata.');

runNames = {'SMD','prediction_inject_count','run_no','fit_attempts', ...
    'fit_successes','fit_success_rate','valid_injection_generations', ...
    'effective_injected_candidates','first_valid_fit_generation', ...
    'first_injection_generation','design_condition_median', ...
    'hessian_evaluable_attempts','hessian_negative_definite_attempts', ...
    'hessian_negative_definite_rate','dominant_fit_failure_reason'};
runRows = cell(height(runs),numel(runNames));
for rowIndex = 1:height(runs)
    matFile = portableRunPath(outputRoot,char(runs.mat_file(rowIndex)));
    loaded = load(matFile,'injectionLog','BI');
    log = loaded.injectionLog;
    startGeneration = loaded.BI.predictionInjection.startGeneration;
    attempts = [log.generation] >= startGeneration & ...
        ~strcmp({log.next_prediction_reason},'disabled_A0');
    valid = [log.next_prediction_valid];
    fitAttempts = sum(attempts);
    fitSuccesses = sum(valid(attempts));
    if fitAttempts == 0
        fitRate = NaN;
    else
        fitRate = fitSuccesses / fitAttempts;
    end
    failedReasons = {log(attempts & ~valid).next_prediction_reason};
    [designConditions,hessianEvaluable,hessianNegativeDefinite] = ...
        collectModelDiagnostics(log,attempts);
    validFitIndices = find(attempts & valid,1,'first');
    injectionIndices = find([log.effective_injected_count] > 0,1,'first');
    if isempty(validFitIndices)
        firstValidFitGeneration = NaN;
    else
        firstValidFitGeneration = log(validFitIndices).generation;
    end
    if isempty(injectionIndices)
        firstInjectionGeneration = NaN;
    else
        firstInjectionGeneration = log(injectionIndices).generation;
    end
    if hessianEvaluable == 0
        hessianRate = NaN;
    else
        hessianRate = hessianNegativeDefinite / hessianEvaluable;
    end
    runRows(rowIndex,:) = {runs.SMD(rowIndex), ...
        runs.prediction_inject_count(rowIndex),runs.run_no(rowIndex), ...
        fitAttempts,fitSuccesses,fitRate, ...
        sum([log.effective_injected_count] > 0), ...
        sum([log.effective_injected_count]),firstValidFitGeneration, ...
        firstInjectionGeneration,finiteMedian(designConditions), ...
        hessianEvaluable,hessianNegativeDefinite,hessianRate, ...
        dominantReason(failedReasons)};
end
fitRuns = cell2table(runRows,'VariableNames',runNames);
if writeOutputs
    writetable(fitRuns,fullfile(outputRoot,'prediction_fit_runs.csv'));
end

summaryNames = {'SMD','prediction_inject_count','runs', ...
    'fit_attempts_total','fit_successes_total','fit_success_rate', ...
    'valid_injection_generations_median', ...
    'effective_injected_candidates_median', ...
    'first_valid_fit_generation_median', ...
    'first_injection_generation_median','design_condition_median', ...
    'hessian_evaluable_attempts_total', ...
    'hessian_negative_definite_attempts_total', ...
    'hessian_negative_definite_rate', ...
    'dominant_fit_failure_reason'};
summaryRows = cell(0,numel(summaryNames));
for smdNo = unique(fitRuns.SMD)'
    for K = unique(fitRuns.prediction_inject_count)'
        subset = fitRuns(fitRuns.SMD == smdNo & ...
            fitRuns.prediction_inject_count == K,:);
        if isempty(subset), continue; end
        attempts = sum(subset.fit_attempts);
        successes = sum(subset.fit_successes);
        if attempts == 0
            rate = NaN;
        else
            rate = successes / attempts;
        end
        hessianEvaluable = sum(subset.hessian_evaluable_attempts);
        hessianNegativeDefinite = sum(subset.hessian_negative_definite_attempts);
        if hessianEvaluable == 0
            hessianRate = NaN;
        else
            hessianRate = hessianNegativeDefinite / hessianEvaluable;
        end
        summaryRows(end+1,:) = {smdNo,K,height(subset),attempts,successes, ...
            rate,median(subset.valid_injection_generations), ...
            median(subset.effective_injected_candidates), ...
            finiteMedian(subset.first_valid_fit_generation), ...
            finiteMedian(subset.first_injection_generation), ...
            finiteMedian(subset.design_condition_median), ...
            hessianEvaluable,hessianNegativeDefinite,hessianRate, ...
            dominantReason(subset.dominant_fit_failure_reason)}; %#ok<AGROW>
    end
end
fitSummary = cell2table(summaryRows,'VariableNames',summaryNames);
if writeOutputs
    writetable(fitSummary,fullfile(outputRoot,'prediction_fit_summary.csv'));
    save(fullfile(outputRoot,'prediction_fit_statistics.mat'),'fitRuns','fitSummary');
end
end

function [designConditions,hessianEvaluable,hessianNegativeDefinite] = ...
    collectModelDiagnostics(log,attempts)
% Read existing model states only.  This is offline analysis and never
% refits a surface or evaluates an SMD objective.
designConditions = zeros(0,1);
hessianEvaluable = 0;
hessianNegativeDefinite = 0;
indices = find(attempts);
for index = indices(:)'
    if ~isfield(log(index),'next_prediction_model')
        continue;
    end
    model = log(index).next_prediction_model;
    if isempty(model) || ~isstruct(model), continue; end
    if isfield(model,'designCondition') && isfinite(model.designCondition)
        designConditions(end+1,1) = model.designCondition; %#ok<AGROW>
    end
    if ~isfield(model,'hessianEigenvalues'), continue; end
    eigenvalues = model.hessianEigenvalues;
    if isempty(eigenvalues) || any(~isfinite(eigenvalues(:))), continue; end
    hessianEvaluable = hessianEvaluable + 1;
    if all(real(eigenvalues(:)) < 0)
        hessianNegativeDefinite = hessianNegativeDefinite + 1;
    end
end
end

function value = finiteMedian(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = median(values); end
end

function reason = dominantReason(reasons)
if isempty(reasons)
    reason = 'none';
    return;
end
if iscell(reasons)
    reasons = reasons(~cellfun(@isempty,reasons));
else
    reasons = cellstr(reasons);
end
if isempty(reasons)
    reason = 'none';
    return;
end
[uniqueReasons,~,indices] = unique(reasons);
counts = accumarray(indices,1);
[~,best] = max(counts);
reason = uniqueReasons{best};
end

function localPath = portableRunPath(outputRoot,storedPath)
if exist(storedPath,'file') == 2
    localPath = storedPath;
    return;
end
storedPath = strrep(storedPath,'/',filesep);
storedPath = strrep(storedPath,'\\',filesep);
localPath = fullfile(outputRoot,storedPath);
assert(exist(localPath,'file') == 2,'Missing run MAT file: %s',localPath);
end
