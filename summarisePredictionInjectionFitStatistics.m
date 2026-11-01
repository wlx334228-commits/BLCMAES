function [fitRuns,fitSummary] = summarisePredictionInjectionFitStatistics(outputRoot)
%SUMMARISEPREDICTIONINJECTIONFITSTATISTICS Summarise model-fit outcomes.
% This is offline bookkeeping only.  It reads each saved injection log and
% adds no upper/lower evaluation or random-number call to BL-CMA-ES.

runFile = fullfile(outputRoot,'injection_runs.csv');
assert(exist(runFile,'file') == 2,'Missing run table: %s',runFile);
runs = readtable(runFile);
required = {'SMD','prediction_inject_count','run_no','mat_file'};
assert(all(ismember(required,runs.Properties.VariableNames)), ...
    'Run table does not contain prediction-injection metadata.');

runNames = {'SMD','prediction_inject_count','run_no','fit_attempts', ...
    'fit_successes','fit_success_rate','valid_injection_generations', ...
    'effective_injected_candidates','dominant_fit_failure_reason'};
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
    runRows(rowIndex,:) = {runs.SMD(rowIndex), ...
        runs.prediction_inject_count(rowIndex),runs.run_no(rowIndex), ...
        fitAttempts,fitSuccesses,fitRate, ...
        sum([log.effective_injected_count] > 0), ...
        sum([log.effective_injected_count]),dominantReason(failedReasons)};
end
fitRuns = cell2table(runRows,'VariableNames',runNames);
writetable(fitRuns,fullfile(outputRoot,'prediction_fit_runs.csv'));

summaryNames = {'SMD','prediction_inject_count','runs', ...
    'fit_attempts_total','fit_successes_total','fit_success_rate', ...
    'valid_injection_generations_median', ...
    'effective_injected_candidates_median', ...
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
        summaryRows(end+1,:) = {smdNo,K,height(subset),attempts,successes, ...
            rate,median(subset.valid_injection_generations), ...
            median(subset.effective_injected_candidates), ...
            dominantReason(subset.dominant_fit_failure_reason)}; %#ok<AGROW>
    end
end
fitSummary = cell2table(summaryRows,'VariableNames',summaryNames);
writetable(fitSummary,fullfile(outputRoot,'prediction_fit_summary.csv'));
save(fullfile(outputRoot,'prediction_fit_statistics.mat'),'fitRuns','fitSummary');
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
