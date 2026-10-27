function [matchedRuns,matchedSummary] = analysePredictionInjectionResults(outputRoot)
%ANALYSEPREDICTIONINJECTIONRESULTS Compare paired runs at a common UFE point.
% The common budget for one A0/variant pair is min(final UFE_A0, final
% UFE_variant). The closest-to-u_fopt value among the candidates evaluated
% up to that point is carried forward; no objective value is re-evaluated.

if nargin < 1 || isempty(outputRoot)
    error('BLCMAES:PredictionInjectionAnalysis', ...
        'Specify a completed prediction-injection result directory.');
end
runFile = fullfile(outputRoot,'injection_runs.csv');
assert(exist(runFile,'file') == 2,'Missing run table: %s',runFile);
runs = readtable(runFile);
assert(all(ismember({'A0','B1','B2'},unique(runs.mode))), ...
    'The result directory must include A0, B1, and B2.');

runNames = {'SMD','mode','run_no','common_UFE', ...
    'A0_carried_upper_error','variant_carried_upper_error','error_delta'};
runRows = cell(0,numel(runNames));
problemNumbers = unique(runs.SMD)';
for smdNo = problemNumbers
    for mode = {'B1','B2'}
        baseline = runs(runs.SMD == smdNo & strcmp(runs.mode,'A0'),:);
        variant = runs(runs.SMD == smdNo & strcmp(runs.mode,mode{1}),:);
        runNumbers = intersect(baseline.run_no,variant.run_no);
        for runNo = runNumbers'
            baseRow = baseline(baseline.run_no == runNo,:);
            variantRow = variant(variant.run_no == runNo,:);
            commonUFE = min(baseRow.UFE,variantRow.UFE);
            baseError = closestErrorAtUFE(char(baseRow.mat_file),commonUFE);
            variantError = closestErrorAtUFE(char(variantRow.mat_file),commonUFE);
            runRows(end+1,:) = {smdNo,mode{1},runNo,commonUFE, ...
                baseError,variantError,variantError-baseError};
        end
    end
end
matchedRuns = cell2table(runRows,'VariableNames',runNames);
writetable(matchedRuns,fullfile(outputRoot,'injection_matched_budget_runs.csv'));

summaryNames = {'SMD','mode','paired_runs','common_UFE_median', ...
    'A0_upper_error_median','variant_upper_error_median', ...
    'error_delta_median','variant_better_runs','variant_better_rate', ...
    'upper_error_p','test_method'};
summaryRows = cell(0,numel(summaryNames));
for smdNo = problemNumbers
    for mode = {'B1','B2'}
        rows = matchedRuns(matchedRuns.SMD == smdNo & ...
            strcmp(matchedRuns.mode,mode{1}),:);
        comparison = pairedWilcoxonSignedRank( ...
            rows.variant_carried_upper_error,rows.A0_carried_upper_error);
        summaryRows(end+1,:) = {smdNo,mode{1},height(rows), ...
            median(rows.common_UFE),median(rows.A0_carried_upper_error), ...
            median(rows.variant_carried_upper_error),median(rows.error_delta), ...
            sum(rows.error_delta < 0),mean(rows.error_delta < 0), ...
            comparison.p,comparison.method};
    end
end
matchedSummary = cell2table(summaryRows,'VariableNames',summaryNames);
writetable(matchedSummary, ...
    fullfile(outputRoot,'injection_matched_budget_summary.csv'));
save(fullfile(outputRoot,'injection_matched_budget.mat'), ...
    'matchedRuns','matchedSummary');
end

function errorValue = closestErrorAtUFE(matFile,budget)
loaded = load(matFile,'injectionLog','BI');
UFE = [loaded.injectionLog.UFE];
index = find(UFE <= budget,1,'last');
if isempty(index)
    error('BLCMAES:PredictionInjectionAnalysis', ...
        'No completed generation at UFE <= %.15g in %s.',budget,matFile);
end
bestError = Inf;
for i = 1:index
    values = [loaded.injectionLog(i).upper_objective_after_true_evaluation(:); ...
        loaded.injectionLog(i).population_UF(:)];
    if loaded.injectionLog(i).external_refinement_upper_evaluated
        values(end+1,1) = loaded.injectionLog(i).external_refinement_UF;
    end
    values = values(isfinite(values));
    if ~isempty(values)
        bestError = min(bestError,min(abs(values-loaded.BI.u_fopt)));
    end
end
errorValue = bestError;
end
