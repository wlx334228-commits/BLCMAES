function [matchedRuns,matchedSummary,pairedTests] = ...
    analysePredictionInjectionResults(outputRoot)
%ANALYSEPREDICTIONINJECTIONRESULTS Paired common-UFE analysis for K studies.
% Each pair is compared at min(final UFE_reference, final UFE_variant).  The
% best error available by that point is carried forward from real evaluations;
% no objective is called by this offline analysis.

if nargin < 1 || isempty(outputRoot)
    error('BLCMAES:PredictionInjectionAnalysis', ...
        'Specify a completed prediction-injection result directory.');
end
runFile = fullfile(outputRoot,'injection_runs.csv');
assert(exist(runFile,'file') == 2,'Missing run table: %s',runFile);
runs = readtable(runFile);
assert(ismember('prediction_inject_count',runs.Properties.VariableNames), ...
    ['This analysis expects the K=0,...,8 experiment table.  The older ' ...
     'A0/B1/B2 table should be analysed in its original result directory.']);
counts = unique(runs.prediction_inject_count)';
references = referencePairs(counts);

runNames = {'SMD','comparison','reference_mode','variant_mode', ...
    'reference_K','variant_K','run_no','common_UFE', ...
    'reference_carried_upper_error','variant_carried_upper_error','error_delta'};
runRows = cell(0,numel(runNames));
for smdNo = unique(runs.SMD)'
    for referenceIndex = 1:size(references,1)
        referenceK = references{referenceIndex,1};
        comparison = references{referenceIndex,2};
        if referenceK == 0, variantKs = counts(counts > 0);
        else, variantKs = counts(counts > 1); end
        reference = runs(runs.SMD == smdNo & ...
            runs.prediction_inject_count == referenceK,:);
        for variantK = variantKs(:)'
            variant = runs(runs.SMD == smdNo & ...
                runs.prediction_inject_count == variantK,:);
            runNumbers = intersect(reference.run_no,variant.run_no);
            for runNo = runNumbers'
                referenceRow = reference(reference.run_no == runNo,:);
                variantRow = variant(variant.run_no == runNo,:);
                commonUFE = min(referenceRow.UFE,variantRow.UFE);
                referenceError = closestErrorAtUFE(outputRoot, ...
                    char(referenceRow.mat_file),commonUFE);
                variantError = closestErrorAtUFE(outputRoot, ...
                    char(variantRow.mat_file),commonUFE);
                runRows(end+1,:) = {smdNo,comparison,sprintf('K%d',referenceK), ...
                    sprintf('K%d',variantK),referenceK,variantK,runNo,commonUFE, ...
                    referenceError,variantError,variantError-referenceError};
            end
        end
    end
end
matchedRuns = cell2table(runRows,'VariableNames',runNames);
writetable(matchedRuns,fullfile(outputRoot,'injection_matched_budget_runs.csv'));

summaryNames = {'SMD','comparison','reference_mode','variant_mode', ...
    'reference_K','variant_K','paired_runs','common_UFE_median', ...
    'reference_upper_error_median','variant_upper_error_median', ...
    'error_delta_median','variant_better_runs','variant_better_rate'};
summaryRows = cell(0,numel(summaryNames));
testNames = [summaryNames,{'upper_error_p','upper_error_p_holm', ...
    'test_method','upper_error_advantage'}];
testRows = cell(0,numel(testNames));
for rowIndex = 1:size(references,1)
    referenceK = references{rowIndex,1};
    comparison = references{rowIndex,2};
    if referenceK == 0, variantKs = counts(counts > 0);
    else, variantKs = counts(counts > 1); end
    for smdNo = unique(runs.SMD)'
        for variantK = variantKs(:)'
            rows = matchedRuns(matchedRuns.SMD == smdNo & ...
                strcmp(matchedRuns.comparison,comparison) & ...
                matchedRuns.variant_K == variantK,:);
            if isempty(rows), continue; end
            base = {smdNo,comparison,sprintf('K%d',referenceK), ...
                sprintf('K%d',variantK),referenceK,variantK,height(rows), ...
                median(rows.common_UFE),median(rows.reference_carried_upper_error), ...
                median(rows.variant_carried_upper_error),median(rows.error_delta), ...
                sum(rows.error_delta < 0),mean(rows.error_delta < 0)};
            summaryRows(end+1,:) = base;
            test = pairedWilcoxonSignedRank( ...
                rows.variant_carried_upper_error,rows.reference_carried_upper_error);
            testRows(end+1,:) = [base,{test.p,NaN,test.method,false}];
        end
    end
end
matchedSummary = cell2table(summaryRows,'VariableNames',summaryNames);
pairedTests = cell2table(testRows,'VariableNames',testNames);
if ~isempty(pairedTests)
    pairedTests = applyHolmAndDirection(pairedTests);
end
writetable(matchedSummary, ...
    fullfile(outputRoot,'injection_matched_budget_summary.csv'));
writetable(pairedTests, ...
    fullfile(outputRoot,'injection_matched_budget_wilcoxon.csv'));
save(fullfile(outputRoot,'injection_matched_budget.mat'), ...
    'matchedRuns','matchedSummary','pairedTests');
end

function references = referencePairs(counts)
references = {};
if any(counts == 0), references(end+1,:) = {0,'vs_K0'}; end %#ok<AGROW>
if any(counts == 1), references(end+1,:) = {1,'vs_K1'}; end %#ok<AGROW>
end

function tests = applyHolmAndDirection(tests)
for smdNo = unique(tests.SMD)'
    comparisons = unique(tests.comparison(tests.SMD == smdNo));
    for i = 1:numel(comparisons)
        index = tests.SMD == smdNo & ...
            strcmp(tests.comparison,comparisons{i});
        tests.upper_error_p_holm(index) = holmAdjust(tests.upper_error_p(index));
    end
end
tests.upper_error_advantage = tests.upper_error_p_holm < 0.05 & ...
    tests.variant_upper_error_median < tests.reference_upper_error_median;
end

function adjusted = holmAdjust(pValues)
adjusted = NaN(size(pValues));
valid = find(isfinite(pValues));
if isempty(valid), return; end
[orderedValues,order] = sort(pValues(valid));
count = numel(orderedValues);
adjustedOrdered = NaN(count,1);
previous = 0;
for i = 1:count
    previous = max(previous,min(1,(count-i+1)*orderedValues(i)));
    adjustedOrdered(i) = previous;
end
adjusted(valid(order)) = adjustedOrdered;
end

function errorValue = closestErrorAtUFE(outputRoot,matFile,budget)
% New K-ablation tables store result-relative paths so an entire experiment
% directory remains analysable after moving it to another computer.
if exist(matFile,'file') ~= 2
    matFile = fullfile(outputRoot,portableRelativePath(matFile));
end
assert(exist(matFile,'file') == 2,'Missing run MAT file: %s',matFile);
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

function localPath = portableRelativePath(relativePath)
localPath = strrep(relativePath,'/',filesep);
localPath = strrep(localPath,'\',filesep);
end
