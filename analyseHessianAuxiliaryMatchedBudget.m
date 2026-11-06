function [matchedRuns,summary,tests] = analyseHessianAuxiliaryMatchedBudget( ...
    runs,problemNumbers,masterRoot)
%ANALYSEHESSIANAUXILIARYMATCHEDBUDGET Paired common-UFE H1 comparison.
% At each pair's min(final UFE), this reads the carried best error from the
% saved real-evaluation log.  It never evaluates a new upper/lower point.

comparisons = {'A1_vs_A0','A0','A1'; ...
    'A2_vs_A1','A1','A2';'H1_vs_A1','A1','H1';'H1_vs_A2','A2','H1'};
runNames = {'SMD','comparison','reference_experiment','variant_experiment', ...
    'run_no','seed','common_UFE','reference_carried_upper_error', ...
    'variant_carried_upper_error','error_delta'};
rows = cell(0,numel(runNames));
for smdNo = problemNumbers(:)'
    for c = 1:size(comparisons,1)
        reference = runs(runs.SMD == smdNo & ...
            strcmp(runs.experiment_id,comparisons{c,2}),:);
        variant = runs(runs.SMD == smdNo & ...
            strcmp(runs.experiment_id,comparisons{c,3}),:);
        [reference,variant] = alignPairs(reference,variant,smdNo,comparisons{c,1});
        for i = 1:height(reference)
            budget = min(reference.UFE(i),variant.UFE(i));
            referenceError = carriedError(reference(i,:),budget,masterRoot);
            variantError = carriedError(variant(i,:),budget,masterRoot);
            rows(end+1,:) = {smdNo,comparisons{c,1},comparisons{c,2}, ...
                comparisons{c,3},reference.run_no(i),reference.seed(i),budget, ...
                referenceError,variantError,variantError-referenceError}; %#ok<AGROW>
        end
    end
end
matchedRuns = cell2table(rows,'VariableNames',runNames);

summaryNames = {'SMD','comparison','reference_experiment','variant_experiment', ...
    'paired_runs','common_UFE_median','reference_upper_error_median', ...
    'variant_upper_error_median','error_delta_median', ...
    'variant_better_runs','variant_better_rate'};
testNames = [summaryNames,{'upper_error_p','upper_error_p_holm', ...
    'test_method','upper_error_advantage'}];
summaryRows = cell(0,numel(summaryNames)); testRows = cell(0,numel(testNames));
for smdNo = problemNumbers(:)'
    for c = 1:size(comparisons,1)
        subset = matchedRuns(matchedRuns.SMD == smdNo & ...
            strcmp(matchedRuns.comparison,comparisons{c,1}),:);
        base = {smdNo,comparisons{c,1},comparisons{c,2},comparisons{c,3}, ...
            height(subset),median(subset.common_UFE), ...
            median(subset.reference_carried_upper_error), ...
            median(subset.variant_carried_upper_error),median(subset.error_delta), ...
            sum(subset.error_delta < 0),mean(subset.error_delta < 0)};
        summaryRows(end+1,:) = base; %#ok<AGROW>
        test = pairedWilcoxonSignedRank(subset.variant_carried_upper_error, ...
            subset.reference_carried_upper_error);
        testRows(end+1,:) = [base,{test.p,NaN,test.method,false}]; %#ok<AGROW>
    end
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

function value = carriedError(row,budget,masterRoot)
resultRoot = char(row.result_root);
if ~isAbsolutePath(resultRoot)
    resultRoot = fullfile(masterRoot,portableRelative(resultRoot));
end
matFile = fullfile(resultRoot,portableRelative(char(row.mat_file)));
assert(exist(matFile,'file') == 2,'Missing run MAT file: %s',matFile);
loaded = load(matFile,'injectionLog','BI');
UFE = [loaded.injectionLog.UFE];
last = find(UFE <= budget,1,'last');
assert(~isempty(last),'No generation at UFE <= %.15g in %s.',budget,matFile);
value = Inf;
for generation = 1:last
    UF = [loaded.injectionLog(generation).upper_objective_after_true_evaluation(:); ...
        loaded.injectionLog(generation).population_UF(:)];
    if loaded.injectionLog(generation).external_refinement_upper_evaluated
        UF(end+1,1) = loaded.injectionLog(generation).external_refinement_UF;
    end
    UF = UF(isfinite(UF));
    if ~isempty(UF), value = min(value,min(abs(UF-loaded.BI.u_fopt))); end
end

function yes = isAbsolutePath(path)
yes = (~isempty(path) && (path(1) == filesep || ...
    (numel(path) >= 3 && path(2) == ':' && (path(3) == '/' || path(3) == '\'))));
end
end

function [reference,variant] = alignPairs(reference,variant,smdNo,comparison)
assert(height(reference) == height(variant) && ~isempty(reference), ...
    'SMD%d %s has unmatched paired data.',smdNo,comparison);
reference = sortrows(reference,'run_no'); variant = sortrows(variant,'run_no');
assert(isequal(reference.run_no,variant.run_no) && ...
    isequal(reference.seed,variant.seed), ...
    'SMD%d %s does not share paired seeds.',smdNo,comparison);
end

function path = portableRelative(path)
path = strrep(path,'/',filesep);
path = strrep(path,'\',filesep);
end

function adjusted = holmAdjust(pValues)
adjusted = NaN(size(pValues)); valid = find(isfinite(pValues));
if isempty(valid), return; end
[ordered,order] = sort(pValues(valid)); previous = 0;
for i = 1:numel(ordered)
    previous = max(previous,min(1,(numel(ordered)-i+1)*ordered(i)));
    adjusted(valid(order(i))) = previous;
end
end
