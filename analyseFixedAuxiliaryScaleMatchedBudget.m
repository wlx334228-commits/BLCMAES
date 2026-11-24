function [matchedRuns,matchedSummary,pairedTests] = ...
    analyseFixedAuxiliaryScaleMatchedBudget( ...
    outputRoot,runs,problemNumbers,comparisons)
%ANALYSEFIXEDAUXILIARYSCALEMATCHEDBUDGET Compare paired variants at a
% common true upper-level evaluation budget.  This is read-only: it walks
% the evaluation logs saved by the four formal branches and never invokes
% an SMD objective or lower-level optimizer.

required = {'SMD','variant','run_no','seed','UFE','mat_file', ...
    'result_directory'};
assert(istable(runs) && all(ismember(required,runs.Properties.VariableNames)), ...
    'The fixed-scale table lacks fields required for common-budget analysis.');

runNames = {'SMD','comparison','reference_variant','variant','run_no', ...
    'seed','common_UFE','reference_carried_upper_error', ...
    'variant_carried_upper_error','error_delta'};
rows = cell(0,numel(runNames));
for smdNo = problemNumbers(:)'
    for comparisonIndex = 1:size(comparisons,1)
        referenceId = comparisons{comparisonIndex,2};
        variantId = comparisons{comparisonIndex,3};
        reference = runs(runs.SMD == smdNo & ...
            strcmp(runs.variant,referenceId),:);
        variant = runs(runs.SMD == smdNo & strcmp(runs.variant,variantId),:);
        [reference,variant] = alignPairs(reference,variant,smdNo, ...
            comparisons{comparisonIndex,1});
        for i = 1:height(reference)
            commonUFE = min(reference.UFE(i),variant.UFE(i));
            referenceError = bestErrorAtUFE(outputRoot,reference(i,:),commonUFE);
            variantError = bestErrorAtUFE(outputRoot,variant(i,:),commonUFE);
            rows(end+1,:) = {smdNo,comparisons{comparisonIndex,1}, ...
                referenceId,variantId,reference.run_no(i),reference.seed(i), ...
                commonUFE,referenceError,variantError,variantError-referenceError}; %#ok<AGROW>
        end
    end
end
matchedRuns = cell2table(rows,'VariableNames',runNames);
[matchedSummary,pairedTests] = summariseMatchedRuns(matchedRuns);
writetable(matchedRuns, ...
    fullfile(outputRoot,'fixed_scale_matched_budget_runs.csv'));
writetable(matchedSummary, ...
    fullfile(outputRoot,'fixed_scale_matched_budget_summary.csv'));
writetable(pairedTests, ...
    fullfile(outputRoot,'fixed_scale_matched_budget_wilcoxon.csv'));
save(fullfile(outputRoot,'fixed_scale_matched_budget.mat'), ...
    'matchedRuns','matchedSummary','pairedTests');
end

function [reference,variant] = alignPairs(reference,variant,smdNo,comparison)
assert(height(reference) == height(variant) && ~isempty(reference), ...
    'SMD%d %s has unmatched paired rows.',smdNo,comparison);
reference = sortrows(reference,'run_no');
variant = sortrows(variant,'run_no');
assert(isequal(reference.run_no,variant.run_no) && ...
    isequal(reference.seed,variant.seed), ...
    'SMD%d %s does not have identical paired seeds.',smdNo,comparison);
end

function errorValue = bestErrorAtUFE(outputRoot,row,budget)
matFile = fullfile(outputRoot,char(row.result_directory), ...
    portableRelativePath(char(row.mat_file)));
assert(exist(matFile,'file') == 2,'Missing run log: %s',matFile);
loaded = load(matFile,'injectionLog','BI');
completedGeneration = find([loaded.injectionLog.UFE] <= budget,1,'last');
assert(~isempty(completedGeneration), ...
    'No completed generation at UFE <= %.15g in %s.',budget,matFile);
errorValue = Inf;
for generation = 1:completedGeneration
    log = loaded.injectionLog(generation);
    values = [log.upper_objective_after_true_evaluation(:);log.population_UF(:)];
    if log.external_refinement_upper_evaluated
        values(end+1,1) = log.external_refinement_UF; %#ok<AGROW>
    end
    values = values(isfinite(values));
    if ~isempty(values)
        errorValue = min(errorValue,min(abs(values-loaded.BI.u_fopt)));
    end
end
assert(isfinite(errorValue),'No finite upper evaluation found in %s.',matFile);
end

function [summary,tests] = summariseMatchedRuns(runs)
summaryNames = {'SMD','comparison','reference_variant','variant', ...
    'paired_runs','common_UFE_median','reference_upper_error_median', ...
    'variant_upper_error_median','error_delta_median', ...
    'variant_better_runs','variant_better_rate'};
testNames = [summaryNames,{'upper_error_p','upper_error_p_holm', ...
    'test_method','upper_error_advantage'}];
summaryRows = cell(0,numel(summaryNames));
testRows = cell(0,numel(testNames));
keys = unique(runs(:,{'SMD','comparison','reference_variant','variant'}), ...
    'rows','stable');
for i = 1:height(keys)
    subset = runs(runs.SMD == keys.SMD(i) & ...
        strcmp(runs.comparison,keys.comparison{i}) & ...
        strcmp(runs.reference_variant,keys.reference_variant{i}) & ...
        strcmp(runs.variant,keys.variant{i}),:);
    base = {keys.SMD(i),keys.comparison{i},keys.reference_variant{i}, ...
        keys.variant{i},height(subset),median(subset.common_UFE), ...
        median(subset.reference_carried_upper_error), ...
        median(subset.variant_carried_upper_error),median(subset.error_delta), ...
        sum(subset.error_delta < 0),mean(subset.error_delta < 0)};
    summaryRows(end+1,:) = base; %#ok<AGROW>
    test = pairedWilcoxonSignedRank(subset.variant_carried_upper_error, ...
        subset.reference_carried_upper_error);
    testRows(end+1,:) = [base,{test.p,NaN,test.method,false}]; %#ok<AGROW>
end
summary = cell2table(summaryRows,'VariableNames',summaryNames);
tests = cell2table(testRows,'VariableNames',testNames);
for smdNo = unique(tests.SMD)'
    index = tests.SMD == smdNo;
    tests.upper_error_p_holm(index) = holmAdjust(tests.upper_error_p(index));
end
tests.upper_error_advantage = tests.upper_error_p_holm < 0.05 & ...
    tests.variant_upper_error_median < tests.reference_upper_error_median;
end

function adjusted = holmAdjust(pValues)
adjusted = NaN(size(pValues));
valid = find(isfinite(pValues));
if isempty(valid), return; end
[ordered,order] = sort(pValues(valid));
adjustedOrdered = NaN(numel(ordered),1);
previous = 0;
for i = 1:numel(ordered)
    previous = max(previous,min(1,(numel(ordered)-i+1)*ordered(i)));
    adjustedOrdered(i) = previous;
end
adjusted(valid(order)) = adjustedOrdered;
end

function localPath = portableRelativePath(relativePath)
localPath = strrep(relativePath,'/',filesep);
localPath = strrep(localPath,char(92),filesep);
end
