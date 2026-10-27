function [summary,pairedTests,runs] = runSMDPredictionAblation( ...
    outputRoot,numberOfRuns,problemNumbers,modes)
%RUNSMDPREDICTIONABLATION Paired A0--A4 experiments for the diagnosis.
% Defaults to SMD1,2,3,6,8; ten seeds; and A0--A4.  All modes reset the
% global RNG to the same seed. A4 uses its private stream for the two-member
% selection, so it does not perturb the CMA-ES/lower-search random sequence.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_prediction_ablation', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 10; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = [1 2 3 6 8]; end
if nargin < 4 || isempty(modes), modes = {'A0','A1','A2','A3','A4'}; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','>=',1});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'Duplicate problems are not allowed.');
modes = normaliseModes(modes);
assert(any(strcmp(modes,'A0')),'A0 is required as the paired baseline.');
assert(exist(outputRoot,'dir') ~= 7 && exist(outputRoot,'file') ~= 2, ...
    'Choose a fresh output directory: %s',outputRoot);
mkdir(outputRoot);

config = struct('problemNumbers',problemNumbers,'numberOfRuns',numberOfRuns, ...
    'modes',{modes},'totalDim',5,'upperDim',2,'lowerDim',3, ...
    'UmaxFEs',2500,'UmaxImprFEs',350,'LmaxFEs',250, ...
    'LmaxImprFEs',25,'weight',0.1,'startGeneration',10, ...
    'historyGenerations',10,'sampleCount',40,'partialCount',2, ...
    'consistencyCheckEnabled',true);
save(fullfile(outputRoot,'experiment_config.mat'),'config');
copySourceSnapshot(projectRoot,outputRoot);

runColumns = {'SMD','mode','run_no','seed','UF','original_F','upper_error', ...
    'upper_success','UFE','LFE_total','generations', ...
    'first_hit_UFE','first_hit_LFE','nominal_budget_carried_error', ...
    'correction_selected_total','correction_applied_total', ...
    'corrected_parent_total','parent_total','corrected_parent_ratio', ...
    'median_dispersion_ratio','abnormal_correction_generations', ...
    'boundary_repairs','consistency_passed','termination_reason', ...
    'runtime_seconds','mat_file'};
runRows = cell(0,numel(runColumns));
for smdNo = problemNumbers(:)'
    baseBI = getBLOPinfo('SMD',smdNo,config.totalDim);
    assert(baseBI.u_dim == config.upperDim && baseBI.l_dim == config.lowerDim, ...
        'This ablation is defined for (2+3) only.');
    baseBI.UmaxFEs = config.UmaxFEs;
    baseBI.UmaxImprFEs = config.UmaxImprFEs;
    baseBI.LmaxFEs = config.LmaxFEs;
    baseBI.LmaxImprFEs = config.LmaxImprFEs;
    baseBI.u_N = 50;
    baseBI.l_N = 50;
    for runNo = 1:numberOfRuns
        seed = (smdNo-1)*numberOfRuns + runNo;
        for modeIndex = 1:numel(modes)
            mode = modes{modeIndex};
            BI = baseBI;
            BI.enablePredictionCorrection = false;
            BI.enableIndividualPredictionCorrection = ~strcmp(mode,'A0');
            BI.individualPredictionCorrection = struct( ...
                'mode',mode,'logEnabled',true,'weight',config.weight, ...
                'startGeneration',config.startGeneration, ...
                'historyGenerations',config.historyGenerations, ...
                'sampleCount',config.sampleCount, ...
                'partialCount',config.partialCount, ...
                'partialSelectionSeed',100000 + seed, ...
                'consistencyCheckEnabled',config.consistencyCheckEnabled);
            rng(seed,'twister');
            rngStateBefore = rng;
            timer = tic;
            ins = BLCMAES(BI);
            runtimeSeconds = toc(timer);
            rngStateAfter = rng;
            individualPredictionLog = ins.individualPredictionLog;
            metrics = extractRunMetrics(ins,individualPredictionLog,BI);
            metadata = struct('problem',BI.fn,'mode',mode,'runNo',runNo, ...
                'seed',seed,'rngStateBefore',rngStateBefore, ...
                'rngStateAfter',rngStateAfter,'runtimeSeconds',runtimeSeconds, ...
                'UFMeaning','UF = -original upper objective; larger is better', ...
                'samplingPolicy','All modes reset global rng(seed,''twister'').', ...
                'A4Policy',['Two transformed members use a private seeded ' ...
                    'RandStream and do not consume global CMA-ES randomness.']);
            runDir = fullfile(outputRoot,sprintf('SMD%d',smdNo),'DU2_DL3',mode);
            if ~exist(runDir,'dir'), mkdir(runDir); end
            matFile = fullfile(runDir, ...
                sprintf('SMD%d_DU2_DL3_%s_run%02d.mat',smdNo,mode,runNo));
            save(matFile,'ins','individualPredictionLog','BI','metadata','metrics','-v7');
            runRows(end+1,:) = {smdNo,mode,runNo,seed,ins.UF,-ins.UF, ...
                metrics.upperError,metrics.upperSuccess,ins.UFEs,ins.LFEs, ...
                metrics.generations,metrics.firstHitUFE,metrics.firstHitLFE, ...
                metrics.nominalBudgetCarriedError,metrics.selectedTotal, ...
                metrics.appliedTotal,metrics.correctedParentTotal, ...
                metrics.parentTotal,metrics.correctedParentRatio, ...
                metrics.medianDispersionRatio,metrics.abnormalGenerations, ...
                metrics.boundaryRepairs,metrics.consistencyPassed, ...
                metrics.terminationReason,runtimeSeconds,matFile};
            runs = cell2table(runRows,'VariableNames',runColumns);
            save(fullfile(outputRoot,'ablation_runs.mat'),'runs','config');
            writetable(runs,fullfile(outputRoot,'ablation_runs.csv'));
            fprintf(['SMD%d run %02d %s err=%.3g UFE=%d LFE=%d ' ...
                'selected=%d parents=%d consistency=%d\n'], ...
                smdNo,runNo,mode,metrics.upperError,ins.UFEs,ins.LFEs, ...
                metrics.selectedTotal,metrics.correctedParentTotal, ...
                metrics.consistencyPassed);
        end
    end
end

[summary,pairedTests] = summariseAblationRuns(runs,problemNumbers,modes);
save(fullfile(outputRoot,'ablation_summary.mat'),'summary','pairedTests','runs','config');
writetable(summary,fullfile(outputRoot,'ablation_summary.csv'));
writetable(pairedTests,fullfile(outputRoot,'ablation_paired_wilcoxon.csv'));
disp(summary);
disp(pairedTests);
fprintf('Prediction ablation saved in %s\n',outputRoot);
end

function modes = normaliseModes(modes)
if ischar(modes), modes = {modes}; end
assert(iscell(modes),'modes must be a character vector or cell array.');
for i = 1:numel(modes)
    if isstring(modes{i}), modes{i} = char(modes{i}); end
    modes{i} = upper(strtrim(modes{i}));
end
assert(all(ismember(modes,{'A0','A1','A2','A3','A4'})), ...
    'modes must only contain A0--A4.');
assert(numel(unique(modes)) == numel(modes),'Duplicate modes are not allowed.');
end

function copySourceSnapshot(projectRoot,outputRoot)
sourceDir = fullfile(outputRoot,'source_snapshot');
mkdir(sourceDir);
sourceFiles = {'BLCMAES.m','predictionCorrectionOptions.m', ...
    'individualPredictionCorrectionOptions.m','fitLocalUpperQuadratic.m', ...
    'applyIndividualPredictionCorrection.m', ...
    'recordIndividualPredictionEvaluationInput.m', ...
    'recordIndividualPredictionTrueEvaluation.m', ...
    'initializeIndividualPredictionState.m', ...
    'finalizeIndividualPredictionCorrection.m', ...
    'appendIndividualPredictionLog.m','pairedWilcoxonSignedRank.m', ...
    'runSMDPredictionAblation.m','summarisePredictionAblationPrecision.m', ...
    'testPredictionAblation.m'};
for i = 1:numel(sourceFiles)
    copyfile(fullfile(projectRoot,sourceFiles{i}),sourceDir);
end
end

function metrics = extractRunMetrics(ins,log,BI)
metrics = struct();
metrics.upperError = abs(ins.UF-BI.u_fopt);
metrics.upperSuccess = metrics.upperError < BI.u_ftol;
metrics.generations = numel(log);
bestErrors = abs([log.best_so_far_UF]-BI.u_fopt);
firstHit = find(bestErrors < BI.u_ftol,1);
metrics.firstHitUFE = NaN;
metrics.firstHitLFE = NaN;
if ~isempty(firstHit)
    metrics.firstHitUFE = log(firstHit).UFE;
    metrics.firstHitLFE = log(firstHit).LFE_total;
end
% A terminated run cannot be evaluated after stopping. Its final best is
% therefore carried forward to the nominal UFE budget for a fair, explicit
% fixed-budget summary without adding hidden evaluations.
metrics.nominalBudgetCarriedError = bestErrors(end);
metrics.selectedTotal = sum([log.correction_selected_count]);
metrics.appliedTotal = sum([log.correction_applied_count]);
parentCounts = arrayfun(@(item)numel(item.selected_parent_indices),log);
correctedParentCounts = arrayfun(@(item) ...
    sum(item.selected_parent_is_transformed),log);
metrics.parentTotal = sum(parentCounts);
metrics.correctedParentTotal = sum(correctedParentCounts);
metrics.correctedParentRatio = metrics.correctedParentTotal / ...
    max(metrics.parentTotal,1);
dispersion = [log.upper_dispersion_ratio];
dispersion = dispersion(isfinite(dispersion));
if isempty(dispersion), metrics.medianDispersionRatio = NaN;
else, metrics.medianDispersionRatio = median(dispersion); end
metrics.abnormalGenerations = sum([log.abnormal_correction]);
metrics.boundaryRepairs = sum(arrayfun(@(item)sum(item.boundary_repaired),log));
checked = [log.consistency_checked];
passed = [log.consistency_passed];
metrics.consistencyPassed = all(checked) && all(logical(passed));
metrics.terminationReason = log(end).termination_reason;
end

function [summary,pairedTests] = summariseAblationRuns(runs,problemNumbers,modes)
summaryColumns = {'SMD','mode','runs','successes','success_rate', ...
    'error_median','error_q1','error_q3','error_IQR', ...
    'UFE_median','UFE_q1','UFE_q3','UFE_IQR', ...
    'LFE_median','LFE_q1','LFE_q3','LFE_IQR', ...
    'first_hit_UFE_median','first_hit_LFE_median', ...
    'nominal_budget_error_median','selected_total_median', ...
    'applied_total_median','corrected_parent_ratio_median', ...
    'dispersion_ratio_median','abnormal_run_count', ...
    'consistency_successes'};
summaryRows = cell(0,numel(summaryColumns));
for smdNo = problemNumbers(:)'
    for modeIndex = 1:numel(modes)
        mode = modes{modeIndex};
        subset = runs(runs.SMD == smdNo & strcmp(runs.mode,mode),:);
        [errorMedian,errorQ1,errorQ3] = medianIQR(subset.upper_error);
        [ufeMedian,ufeQ1,ufeQ3] = medianIQR(subset.UFE);
        [lfeMedian,lfeQ1,lfeQ3] = medianIQR(subset.LFE_total);
        [firstUFE,~,~] = medianIQR(subset.first_hit_UFE);
        [firstLFE,~,~] = medianIQR(subset.first_hit_LFE);
        [budgetError,~,~] = medianIQR(subset.nominal_budget_carried_error);
        [selectedTotal,~,~] = medianIQR(subset.correction_selected_total);
        [appliedTotal,~,~] = medianIQR(subset.correction_applied_total);
        [parentRatio,~,~] = medianIQR(subset.corrected_parent_ratio);
        [dispersionRatio,~,~] = medianIQR(subset.median_dispersion_ratio);
        summaryRows(end+1,:) = {smdNo,mode,height(subset), ...
            sum(subset.upper_success),mean(subset.upper_success), ...
            errorMedian,errorQ1,errorQ3,errorQ3-errorQ1, ...
            ufeMedian,ufeQ1,ufeQ3,ufeQ3-ufeQ1, ...
            lfeMedian,lfeQ1,lfeQ3,lfeQ3-lfeQ1, ...
            firstUFE,firstLFE,budgetError,selectedTotal,appliedTotal, ...
            parentRatio,dispersionRatio,sum(subset.abnormal_correction_generations > 0), ...
            sum(subset.consistency_passed)};
    end
end
summary = cell2table(summaryRows,'VariableNames',summaryColumns);

pairedColumns = {'SMD','mode','paired_runs','error_p','error_method', ...
    'UFE_p','UFE_method','LFE_p','LFE_method','budget_error_p', ...
    'budget_error_method'};
pairedRows = cell(0,numel(pairedColumns));
for smdNo = problemNumbers(:)'
    baseline = runs(runs.SMD == smdNo & strcmp(runs.mode,'A0'),:);
    for modeIndex = 1:numel(modes)
        mode = modes{modeIndex};
        if strcmp(mode,'A0'), continue; end
        variant = runs(runs.SMD == smdNo & strcmp(runs.mode,mode),:);
        [~,baseIndex,variantIndex] = intersect(baseline.run_no,variant.run_no);
        errorTest = pairedWilcoxonSignedRank( ...
            variant.upper_error(variantIndex),baseline.upper_error(baseIndex));
        ufeTest = pairedWilcoxonSignedRank( ...
            variant.UFE(variantIndex),baseline.UFE(baseIndex));
        lfeTest = pairedWilcoxonSignedRank( ...
            variant.LFE_total(variantIndex),baseline.LFE_total(baseIndex));
        budgetTest = pairedWilcoxonSignedRank( ...
            variant.nominal_budget_carried_error(variantIndex), ...
            baseline.nominal_budget_carried_error(baseIndex));
        pairedRows(end+1,:) = {smdNo,mode,numel(baseIndex),errorTest.p, ...
            errorTest.method,ufeTest.p,ufeTest.method,lfeTest.p, ...
            lfeTest.method,budgetTest.p,budgetTest.method};
    end
end
pairedTests = cell2table(pairedRows,'VariableNames',pairedColumns);
end

function [middle,q1,q3] = medianIQR(values)
values = values(isfinite(values));
if isempty(values)
    middle = NaN; q1 = NaN; q3 = NaN;
    return;
end
values = sort(values(:));
middle = interpolatedPercentile(values,50);
q1 = interpolatedPercentile(values,25);
q3 = interpolatedPercentile(values,75);
end

function value = interpolatedPercentile(sortedValues,percentage)
position = 1 + (numel(sortedValues)-1) * percentage / 100;
lower = floor(position);
upper = ceil(position);
if lower == upper
    value = sortedValues(lower);
else
    value = sortedValues(lower) + (position-lower) * ...
        (sortedValues(upper)-sortedValues(lower));
end
end
