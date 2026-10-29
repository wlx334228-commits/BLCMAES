function [summary,pairedTests,runs] = runSMDPredictionInjection( ...
    outputRoot,numberOfRuns,problemNumbers,predictionInjectCounts)
%RUNSMDPREDICTIONINJECTION K=0,...,8 prediction-candidate injection study.
% K=0 evaluates eight normal CMA-ES candidates.  For valid K>=1 prediction,
% the final K slots are one direct prediction and K-1 private-stream samples
% from the auxiliary distribution centred on that prediction.
%
% Examples
%   runSMDPredictionInjection                         % SMD1--12, K=0:8, 20 runs
%   runSMDPredictionInjection([],1,1,[0 1 2 4 8])     % small validation batch
%   runSMDPredictionInjection(resultDir,20,[1 3 5],0:4) % resumable subset

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_prediction_injection_K', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 20; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = 1:12; end
if nargin < 4 || isempty(predictionInjectCounts)
    predictionInjectCounts = 0:8;
end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','>=',1});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',12});
assert(numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'Duplicate problems are not allowed.');
predictionInjectCounts = normalisePredictionInjectCounts(predictionInjectCounts);

[config,isNewDirectory] = initialiseExperimentDirectory( ...
    outputRoot,problemNumbers,numberOfRuns,predictionInjectCounts,projectRoot);
runColumns = runColumnNames();
runs = loadOrCreateRunTable(outputRoot,runColumns);

for smdNo = problemNumbers(:)'
    baseBI = getBLOPinfo('SMD',smdNo,config.totalDim);
    assert(baseBI.u_dim == config.upperDim && baseBI.l_dim == config.lowerDim, ...
        'This experiment is defined for (2+3) only.');
    baseBI.UmaxFEs = config.UmaxFEs;
    baseBI.UmaxImprFEs = config.UmaxImprFEs;
    baseBI.LmaxFEs = config.LmaxFEs;
    baseBI.LmaxImprFEs = config.LmaxImprFEs;
    baseBI.u_N = 50;
    baseBI.l_N = 50;
    for runNo = 1:numberOfRuns
        % Preserve the seed definition from the previous 20-run B1/B2 study.
        seed = (smdNo-1)*numberOfRuns + runNo;
        for K = predictionInjectCounts(:)'
            mode = sprintf('K%d',K);
            % Store a slash-separated relative path in CSV/MAT metadata.  It
            % remains valid when the full project/result folder moves between
            % macOS, Windows and Linux.
            matFileRelative = sprintf('SMD%d/DU2_DL3/%s/SMD%d_DU2_DL3_%s_run%02d.mat', ...
                smdNo,mode,smdNo,mode,runNo);
            runDir = fullfile(outputRoot,sprintf('SMD%d',smdNo),'DU2_DL3',mode);
            if ~exist(runDir,'dir'), mkdir(runDir); end
            matFile = fullfile(outputRoot,portableRelativePath(matFileRelative));
            completed = runs.SMD == smdNo & ...
                runs.prediction_inject_count == K & runs.run_no == runNo;
            if any(completed) && exist(matFile,'file') == 2
                assert(sum(completed) == 1, ...
                    'Duplicate completed row for SMD%d K=%d run %d.',smdNo,K,runNo);
                assert(runs.seed(find(completed,1)) == seed, ...
                    'Existing run uses a different paired seed.');
                fprintf('Skip completed SMD%d K=%d run %02d\n',smdNo,K,runNo);
                continue;
            end
            if any(completed)
                % Replace a stale table row whose MAT file was removed.
                runs(completed,:) = [];
            end

            BI = baseBI;
            BI.enablePredictionCorrection = false;
            BI.predictionCorrection = struct('logEnabled',false);
            BI.enableIndividualPredictionCorrection = false;
            BI.individualPredictionCorrection = struct('mode','A0','logEnabled',false);
            BI.predictionInjection = struct('mode',mode, ...
                'predictionInjectCount',K,'logEnabled',true, ...
                'startGeneration',config.startGeneration, ...
                'historyGenerations',config.historyGenerations, ...
                'sampleCount',config.sampleCount, ...
                'auxiliaryRandomSeed',500000+seed, ...
                'randomSeed',seed, ...
                'consistencyCheckEnabled',config.consistencyCheckEnabled);
            rng(seed,'twister');
            rngStateBefore = rng;
            timer = tic;
            ins = BLCMAES(BI);
            runtimeSeconds = toc(timer);
            rngStateAfter = rng;
            injectionLog = ins.predictionInjectionLog;
            metrics = extractMetrics(ins,injectionLog,BI);
            metadata = struct('problem',BI.fn,'mode',mode, ...
                'predictionInjectCount',K,'runNo',runNo,'seed',seed, ...
                'rngStateBefore',rngStateBefore,'rngStateAfter',rngStateAfter, ...
                'runtimeSeconds',runtimeSeconds, ...
                'UFMeaning','UF = -original upper objective; larger is better', ...
                'samplingPolicy',['All modes reset global rng(seed,''twister'') ' ...
                    'and draw the original lambda normal joint samples first. ' ...
                    'K>=2 auxiliary samples use a private RandStream.'], ...
                'injectionPolicy',describeInjectionPolicy(K));
            save(matFile,'ins','injectionLog','BI','metadata','metrics','-v7');

            row = makeRunRow(smdNo,mode,K,runNo,seed,ins,metrics, ...
                runtimeSeconds,matFileRelative,runColumns);
            runs = [runs;row]; %#ok<AGROW>
            runs = sortrows(runs,{'SMD','prediction_inject_count','run_no'});
            saveRunTable(outputRoot,runs,config);
            fprintf(['SMD%d K=%d run %02d Uerr=%.3g Lerr=%.3g UFE=%d LFE=%d ' ...
                'directParents=%d auxParents=%d normalParents=%d consistent=%d\n'], ...
                smdNo,K,runNo,metrics.upperError,metrics.lowerError, ...
                ins.UFEs,ins.LFEs,metrics.directParents,metrics.auxiliaryParents, ...
                metrics.normalParents,metrics.consistencyPassed);
        end
    end
end

% A resumed subset must not discard statistics for already completed modes or
% problems in the same output directory.
completedProblems = unique(runs.SMD)';
completedCounts = unique(runs.prediction_inject_count)';
[summary,pairedTests] = summariseRuns(runs,completedProblems,completedCounts);
save(fullfile(outputRoot,'injection_summary.mat'), ...
    'summary','pairedTests','runs','config');
writetable(summary,fullfile(outputRoot,'injection_summary.csv'));
writetable(pairedTests,fullfile(outputRoot,'injection_paired_wilcoxon.csv'));
[matchedBudgetRuns,matchedBudgetSummary,matchedBudgetTests] = ...
    analysePredictionInjectionResults(outputRoot); %#ok<NASGU>
if config.generateFigures
    plotPredictionInjectionKResults(outputRoot);
end
if isNewDirectory
    fprintf('Prediction-injection K experiment saved in %s\n',outputRoot);
else
    fprintf('Prediction-injection K experiment resumed in %s\n',outputRoot);
end
end

function [config,isNewDirectory] = initialiseExperimentDirectory( ...
    outputRoot,problemNumbers,numberOfRuns,predictionInjectCounts,projectRoot)
isNewDirectory = exist(outputRoot,'dir') ~= 7;
assert(exist(outputRoot,'file') ~= 2, ...
    'outputRoot is an existing file: %s',outputRoot);
if isNewDirectory
    mkdir(outputRoot);
    config = makeConfig(problemNumbers,numberOfRuns,predictionInjectCounts);
    save(fullfile(outputRoot,'experiment_config.mat'),'config');
    copySourceSnapshot(projectRoot,outputRoot);
    return;
end
configFile = fullfile(outputRoot,'experiment_config.mat');
assert(exist(configFile,'file') == 2, ...
    ['Existing outputRoot has no experiment_config.mat. Choose a new ' ...
     'directory or restore the matching configuration.']);
loaded = load(configFile,'config');
config = loaded.config;
assert(config.totalDim == 5 && config.upperDim == 2 && config.lowerDim == 3, ...
    'Existing outputRoot is not a (2+3) prediction-injection experiment.');
assert(config.numberOfRuns == numberOfRuns, ...
    ['numberOfRuns differs from the existing directory. Keep it at %d so ' ...
     'the paired seed definition remains unchanged.'],config.numberOfRuns);
assert(config.UmaxFEs == 2500 && config.LmaxFEs == 250 && ...
       config.startGeneration == 10 && config.historyGenerations == 10 && ...
       config.sampleCount == 40, ...
    'Existing outputRoot uses incompatible prediction-injection settings.');
assert(all(ismember(predictionInjectCounts,0:8)), ...
    'K must be between zero and eight.');
end

function config = makeConfig(problemNumbers,numberOfRuns,predictionInjectCounts)
config = struct('problemNumbers',problemNumbers, ...
    'numberOfRuns',numberOfRuns, ...
    'predictionInjectCounts',predictionInjectCounts, ...
    'totalDim',5,'upperDim',2,'lowerDim',3, ...
    'UmaxFEs',2500,'UmaxImprFEs',350,'LmaxFEs',250, ...
    'LmaxImprFEs',25,'startGeneration',10,'historyGenerations',10, ...
    'sampleCount',40,'consistencyCheckEnabled',true,'generateFigures',true);
end

function counts = normalisePredictionInjectCounts(counts)
if ischar(counts) || isstring(counts), counts = cellstr(counts); end
if iscell(counts)
    converted = zeros(1,numel(counts));
    for i = 1:numel(counts)
        value = upper(strtrim(char(counts{i})));
        switch value
            case 'A0', converted(i) = 0;
            case 'B1', converted(i) = 1;
            case 'B2', converted(i) = 2;
            otherwise
                parsed = sscanf(value,'K%d');
                assert(isscalar(parsed) && strcmp(sprintf('K%d',parsed),value), ...
                    'Unknown injection mode: %s',value);
                converted(i) = parsed;
        end
    end
    counts = converted;
end
validateattributes(counts,{'numeric'}, ...
    {'real','finite','vector','integer','>=',0,'<=',8});
counts = unique(counts(:)');
end

function names = runColumnNames()
names = {'SMD','mode','prediction_inject_count','run_no','seed','UF','LF', ...
    'upper_error','lower_error','upper_success','lower_success','UFE', ...
    'LFE_total','generations','first_hit_UFE','first_hit_LFE', ...
    'nominal_budget_carried_error','prediction_injected_generations', ...
    'requested_injected_total','effective_injected_total', ...
    'direct_candidate_total','auxiliary_candidate_total', ...
    'normal_candidate_total','direct_parent_total','auxiliary_parent_total', ...
    'normal_parent_total','direct_parent_selection_rate', ...
    'auxiliary_parent_selection_rate','normal_parent_selection_rate', ...
    'direct_fitness_rank_median','auxiliary_fitness_rank_median', ...
    'direct_generation_best_total','auxiliary_generation_best_total', ...
    'direct_new_best_total','auxiliary_new_best_total','new_best_total', ...
    'injected_step_limited_total','boundary_repairs','max_cond_C', ...
    'min_eig_C','numerical_issue_generations','flat_termination', ...
    'consistency_passed','termination_reason','runtime_seconds','mat_file'};
end

function runs = loadOrCreateRunTable(outputRoot,names)
runMatFile = fullfile(outputRoot,'injection_runs.mat');
runCsvFile = fullfile(outputRoot,'injection_runs.csv');
if exist(runMatFile,'file') == 2
    loaded = load(runMatFile,'runs');
    runs = loaded.runs;
elseif exist(runCsvFile,'file') == 2
    runs = readtable(runCsvFile);
else
    runs = cell2table(cell(0,numel(names)),'VariableNames',names);
end
assert(isequal(runs.Properties.VariableNames,names), ...
    ['Existing run table columns do not match this K-ablation version. ' ...
     'Use a new output directory.']);
end

function row = makeRunRow(smdNo,mode,K,runNo,seed,ins,metrics, ...
    runtimeSeconds,matFile,names)
values = {smdNo,mode,K,runNo,seed,ins.UF,ins.LF, ...
    metrics.upperError,metrics.lowerError,metrics.upperSuccess, ...
    metrics.lowerSuccess,ins.UFEs,ins.LFEs,metrics.generations, ...
    metrics.firstHitUFE,metrics.firstHitLFE, ...
    metrics.nominalBudgetCarriedError,metrics.injectedGenerations, ...
    metrics.requestedInjected,metrics.effectiveInjected, ...
    metrics.directCandidates,metrics.auxiliaryCandidates, ...
    metrics.normalCandidates,metrics.directParents,metrics.auxiliaryParents, ...
    metrics.normalParents,metrics.directParentSelectionRate, ...
    metrics.auxiliaryParentSelectionRate,metrics.normalParentSelectionRate, ...
    metrics.directFitnessRankMedian,metrics.auxiliaryFitnessRankMedian, ...
    metrics.directGenerationBests,metrics.auxiliaryGenerationBests, ...
    metrics.directNewBests,metrics.auxiliaryNewBests,metrics.newBests, ...
    metrics.injectedLimitedSteps,metrics.boundaryRepairs,metrics.maxCondC, ...
    metrics.minEigC,metrics.numericalIssueGenerations,metrics.flatTermination, ...
    metrics.consistencyPassed,metrics.terminationReason,runtimeSeconds,matFile};
row = cell2table(values,'VariableNames',names);
end

function saveRunTable(outputRoot,runs,config)
save(fullfile(outputRoot,'injection_runs.mat'),'runs','config');
writetable(runs,fullfile(outputRoot,'injection_runs.csv'));
end

function localPath = portableRelativePath(relativePath)
localPath = strrep(relativePath,'/',filesep);
localPath = strrep(localPath,'\',filesep);
end

function description = describeInjectionPolicy(K)
if K == 0
    description = 'K=0: eight normal CMA-ES candidates; no model fit or injection.';
else
    description = sprintf(['K=%d: %d final normal candidate(s), one direct ' ...
        'quadratic prediction, and %d auxiliary joint sample(s) centred on ' ...
        'the bounded prediction.'],K,8-K,K-1);
end
end

function copySourceSnapshot(projectRoot,outputRoot)
sourceDir = fullfile(outputRoot,'source_snapshot');
mkdir(sourceDir);
files = {'BLCMAES.m','predictionInjectionOptions.m', ...
    'initializePredictionInjectionState.m','applyPredictionInjection.m', ...
    'recordPredictionInjectionEvaluationInput.m', ...
    'recordPredictionInjectionTrueEvaluation.m', ...
    'finalizePredictionInjection.m','appendPredictionInjectionLog.m', ...
    'fitLocalUpperQuadratic.m','pairedWilcoxonSignedRank.m', ...
    'analysePredictionInjectionResults.m','plotPredictionInjectionKResults.m', ...
    'runSMDPredictionInjection.m','testPredictionInjection.m'};
for i = 1:numel(files)
    copyfile(fullfile(projectRoot,files{i}),sourceDir);
end
end

function metrics = extractMetrics(ins,log,BI)
metrics = struct();
metrics.upperError = abs(ins.UF-BI.u_fopt);
metrics.lowerError = abs(ins.LF-BI.l_fopt);
metrics.upperSuccess = metrics.upperError < BI.u_ftol;
metrics.lowerSuccess = metrics.lowerError < BI.l_ftol;
metrics.generations = numel(log);
bestErrors = bestLoggedAccuracyErrors(log,BI.u_fopt);
firstHit = find(bestErrors < BI.u_ftol,1);
metrics.firstHitUFE = NaN;
metrics.firstHitLFE = NaN;
if ~isempty(firstHit)
    metrics.firstHitUFE = log(firstHit).UFE;
    metrics.firstHitLFE = log(firstHit).LFE_total;
end
metrics.nominalBudgetCarriedError = bestErrors(end);
metrics.injectedGenerations = sum([log.effective_injected_count] > 0);
metrics.requestedInjected = sum([log.requested_injected_count]);
metrics.effectiveInjected = sum([log.effective_injected_count]);
metrics.directCandidates = sum(arrayfun(@(item) ...
    sum(strcmp(item.source_labels,'predicted_direct')),log));
metrics.auxiliaryCandidates = sum(arrayfun(@(item) ...
    sum(strcmp(item.source_labels,'predicted_auxiliary')),log));
metrics.normalCandidates = sum(arrayfun(@(item) ...
    sum(strcmp(item.source_labels,'normal')),log));
metrics.directParents = sum([log.direct_parent_count]);
metrics.auxiliaryParents = sum([log.auxiliary_parent_count]);
metrics.normalParents = sum([log.normal_parent_count]);
metrics.directParentSelectionRate = safeRatio(metrics.directParents, ...
    metrics.directCandidates);
metrics.auxiliaryParentSelectionRate = safeRatio(metrics.auxiliaryParents, ...
    metrics.auxiliaryCandidates);
metrics.normalParentSelectionRate = safeRatio(metrics.normalParents, ...
    metrics.normalCandidates);
metrics.directFitnessRankMedian = finiteMedian( ...
    collectLogValues(log,'direct_fitness_ranks'));
metrics.auxiliaryFitnessRankMedian = finiteMedian( ...
    collectLogValues(log,'auxiliary_fitness_ranks'));
metrics.directGenerationBests = sum([log.direct_is_generation_best]);
metrics.auxiliaryGenerationBests = sum([log.auxiliary_is_generation_best]);
metrics.directNewBests = sum(arrayfun(@(item) ...
    item.new_best_so_far && item.direct_is_generation_best,log));
metrics.auxiliaryNewBests = sum(arrayfun(@(item) ...
    item.new_best_so_far && item.auxiliary_is_generation_best,log));
metrics.newBests = sum([log.new_best_so_far]);
metrics.injectedLimitedSteps = sum(arrayfun(@(item) ...
    sum(item.injected_step_limited),log));
metrics.boundaryRepairs = sum(arrayfun(@(item) ...
    sum(item.direct_boundary_repaired) + sum(item.auxiliary_boundary_repaired(:)),log));
conditions = [log.cond_C_after_update];
eigenMins = [log.eig_min_C_after_update];
metrics.maxCondC = finiteMaximum(conditions);
metrics.minEigC = finiteMinimum(eigenMins);
metrics.numericalIssueGenerations = sum(~isfinite(conditions) | ...
    ~isfinite(eigenMins) | eigenMins <= 0);
metrics.flatTermination = strcmp(log(end).termination_reason,'flat_progress');
checked = [log.consistency_checked];
passed = [log.consistency_passed];
metrics.consistencyPassed = all(checked) && all(logical(passed));
metrics.terminationReason = log(end).termination_reason;
end

function values = collectLogValues(log,fieldName)
values = zeros(0,1);
for i = 1:numel(log)
    itemValues = log(i).(fieldName);
    values = [values;itemValues(:)]; %#ok<AGROW>
end
end

function ratio = safeRatio(numerator,denominator)
if denominator == 0, ratio = NaN; else, ratio = numerator / denominator; end
end

function value = finiteMedian(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = median(values); end
end

function value = finiteMaximum(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = max(values); end
end

function value = finiteMinimum(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = min(values); end
end

function errors = bestLoggedAccuracyErrors(log,uFopt)
errors = NaN(1,numel(log));
bestError = Inf;
for i = 1:numel(log)
    values = [log(i).upper_objective_after_true_evaluation(:); ...
        log(i).population_UF(:)];
    if log(i).external_refinement_upper_evaluated
        values(end+1,1) = log(i).external_refinement_UF;
    end
    values = values(isfinite(values));
    if ~isempty(values)
        bestError = min(bestError,min(abs(values-uFopt)));
    end
    errors(i) = bestError;
end
end

function [summary,pairedTests] = summariseRuns(runs,problemNumbers,counts)
summaryNames = {'SMD','mode','prediction_inject_count','runs', ...
    'upper_successes','upper_success_rate','lower_successes','lower_success_rate', ...
    'upper_error_median','upper_error_IQR','lower_error_median','lower_error_IQR', ...
    'UFE_median','UFE_IQR','LFE_median','LFE_IQR', ...
    'first_hit_UFE_median_successful_runs', ...
    'first_hit_LFE_median_successful_runs','budget_error_median', ...
    'prediction_injected_generations_median','requested_injected_total_median', ...
    'effective_injected_total_median','direct_candidate_total_median', ...
    'auxiliary_candidate_total_median','normal_candidate_total_median', ...
    'direct_parent_total_median','auxiliary_parent_total_median', ...
    'normal_parent_total_median','direct_parent_selection_rate_median', ...
    'auxiliary_parent_selection_rate_median','normal_parent_selection_rate_median', ...
    'direct_fitness_rank_median','auxiliary_fitness_rank_median', ...
    'max_cond_C_median','min_eig_C_median','numerical_issue_run_count', ...
    'flat_termination_run_count','boundary_repairs_total','consistency_successes'};
summaryRows = cell(0,numel(summaryNames));
for smdNo = problemNumbers(:)'
    for K = counts(:)'
        mode = sprintf('K%d',K);
        subset = runs(runs.SMD == smdNo & ...
            runs.prediction_inject_count == K,:);
        [upperError,upperIQR] = medianIQR(subset.upper_error);
        [lowerError,lowerIQR] = medianIQR(subset.lower_error);
        [ufe,ufeIQR] = medianIQR(subset.UFE);
        [lfe,lfeIQR] = medianIQR(subset.LFE_total);
        [firstUFE,~] = medianIQR(subset.first_hit_UFE);
        [firstLFE,~] = medianIQR(subset.first_hit_LFE);
        [budgetError,~] = medianIQR(subset.nominal_budget_carried_error);
        [injectedGenerations,~] = medianIQR(subset.prediction_injected_generations);
        [requested,~] = medianIQR(subset.requested_injected_total);
        [effective,~] = medianIQR(subset.effective_injected_total);
        [directCandidates,~] = medianIQR(subset.direct_candidate_total);
        [auxCandidates,~] = medianIQR(subset.auxiliary_candidate_total);
        [normalCandidates,~] = medianIQR(subset.normal_candidate_total);
        [directParents,~] = medianIQR(subset.direct_parent_total);
        [auxParents,~] = medianIQR(subset.auxiliary_parent_total);
        [normalParents,~] = medianIQR(subset.normal_parent_total);
        [directParentRate,~] = medianIQR(subset.direct_parent_selection_rate);
        [auxParentRate,~] = medianIQR(subset.auxiliary_parent_selection_rate);
        [normalParentRate,~] = medianIQR(subset.normal_parent_selection_rate);
        [directRank,~] = medianIQR(subset.direct_fitness_rank_median);
        [auxRank,~] = medianIQR(subset.auxiliary_fitness_rank_median);
        [maxCond,~] = medianIQR(subset.max_cond_C);
        [minEig,~] = medianIQR(subset.min_eig_C);
        summaryRows(end+1,:) = {smdNo,mode,K,height(subset), ...
            sum(subset.upper_success),mean(subset.upper_success), ...
            sum(subset.lower_success),mean(subset.lower_success), ...
            upperError,upperIQR,lowerError,lowerIQR,ufe,ufeIQR,lfe,lfeIQR, ...
            firstUFE,firstLFE,budgetError,injectedGenerations,requested,effective, ...
            directCandidates,auxCandidates,normalCandidates,directParents,auxParents, ...
            normalParents,directParentRate,auxParentRate,normalParentRate, ...
            directRank,auxRank,maxCond,minEig, ...
            sum(subset.numerical_issue_generations > 0), ...
            sum(subset.flat_termination),sum(subset.boundary_repairs), ...
            sum(subset.consistency_passed)};
    end
end
summary = cell2table(summaryRows,'VariableNames',summaryNames);
pairedTests = buildPairedTests(runs,problemNumbers,counts);
end

function tests = buildPairedTests(runs,problemNumbers,counts)
names = {'SMD','comparison','reference_mode','variant_mode', ...
    'reference_K','variant_K','paired_runs','reference_successes', ...
    'variant_successes','reference_success_rate','variant_success_rate', ...
    'upper_error_reference_median','upper_error_variant_median','upper_error_p', ...
    'lower_error_p','UFE_reference_median','UFE_variant_median','UFE_p', ...
    'LFE_reference_median','LFE_variant_median','LFE_p', ...
    'budget_error_reference_median','budget_error_variant_median','budget_error_p', ...
    'test_method','upper_error_p_holm','lower_error_p_holm','UFE_p_holm', ...
    'LFE_p_holm','budget_error_p_holm','upper_error_advantage', ...
    'UFE_advantage','LFE_advantage','budget_error_advantage'};
rows = cell(0,numel(names));
references = {};
if any(counts == 0), references(end+1,:) = {0,'vs_K0'}; end %#ok<AGROW>
if any(counts == 1), references(end+1,:) = {1,'vs_K1'}; end %#ok<AGROW>
for smdNo = problemNumbers(:)'
    for referenceRow = 1:size(references,1)
        referenceK = references{referenceRow,1};
        comparison = references{referenceRow,2};
        if referenceK == 0, variantKs = counts(counts > 0);
        else, variantKs = counts(counts > 1); end
        reference = runs(runs.SMD == smdNo & ...
            runs.prediction_inject_count == referenceK,:);
        for variantK = variantKs(:)'
            variant = runs(runs.SMD == smdNo & ...
                runs.prediction_inject_count == variantK,:);
            [runNos,referenceIndex,variantIndex] = intersect( ...
                reference.run_no,variant.run_no); %#ok<ASGLU>
            if isempty(runNos), continue; end
            referencePair = reference(referenceIndex,:);
            variantPair = variant(variantIndex,:);
            upper = pairedWilcoxonSignedRank( ...
                variantPair.upper_error,referencePair.upper_error);
            lower = pairedWilcoxonSignedRank( ...
                variantPair.lower_error,referencePair.lower_error);
            ufe = pairedWilcoxonSignedRank(variantPair.UFE,referencePair.UFE);
            lfe = pairedWilcoxonSignedRank( ...
                variantPair.LFE_total,referencePair.LFE_total);
            budget = pairedWilcoxonSignedRank( ...
                variantPair.nominal_budget_carried_error, ...
                referencePair.nominal_budget_carried_error);
            rows(end+1,:) = {smdNo,comparison,sprintf('K%d',referenceK), ...
                sprintf('K%d',variantK),referenceK,variantK,height(referencePair), ...
                sum(referencePair.upper_success),sum(variantPair.upper_success), ...
                mean(referencePair.upper_success),mean(variantPair.upper_success), ...
                median(referencePair.upper_error),median(variantPair.upper_error),upper.p, ...
                lower.p,median(referencePair.UFE),median(variantPair.UFE),ufe.p, ...
                median(referencePair.LFE_total),median(variantPair.LFE_total),lfe.p, ...
                median(referencePair.nominal_budget_carried_error), ...
                median(variantPair.nominal_budget_carried_error),budget.p,upper.method, ...
                NaN,NaN,NaN,NaN,NaN,false,false,false,false};
        end
    end
end
tests = cell2table(rows,'VariableNames',names);
if ~isempty(tests), tests = applyHolmAndAdvantages(tests); end
end

function tests = applyHolmAndAdvantages(tests)
rawNames = {'upper_error_p','lower_error_p','UFE_p','LFE_p','budget_error_p'};
for rawIndex = 1:numel(rawNames)
    adjustedName = [rawNames{rawIndex},'_holm'];
    for smdNo = unique(tests.SMD)'
        comparisons = unique(tests.comparison(tests.SMD == smdNo));
        for comparisonIndex = 1:numel(comparisons)
            indices = tests.SMD == smdNo & ...
                strcmp(tests.comparison,comparisons{comparisonIndex});
            tests.(adjustedName)(indices) = holmAdjust(tests.(rawNames{rawIndex})(indices));
        end
    end
end
tests.upper_error_advantage = tests.upper_error_p_holm < 0.05 & ...
    tests.upper_error_variant_median < tests.upper_error_reference_median;
noSuccessLoss = tests.variant_successes >= tests.reference_successes;
tests.UFE_advantage = tests.UFE_p_holm < 0.05 & noSuccessLoss & ...
    tests.UFE_variant_median < tests.UFE_reference_median;
tests.LFE_advantage = tests.LFE_p_holm < 0.05 & noSuccessLoss & ...
    tests.LFE_variant_median < tests.LFE_reference_median;
tests.budget_error_advantage = tests.budget_error_p_holm < 0.05 & ...
    tests.budget_error_variant_median < tests.budget_error_reference_median;
end

function adjusted = holmAdjust(pValues)
adjusted = NaN(size(pValues));
valid = find(isfinite(pValues));
if isempty(valid), return; end
[sorted,order] = sort(pValues(valid));
count = numel(sorted);
adjustedSorted = NaN(count,1);
largestPrevious = 0;
for i = 1:count
    largestPrevious = max(largestPrevious,min(1,(count-i+1)*sorted(i)));
    adjustedSorted(i) = largestPrevious;
end
adjusted(valid(order)) = adjustedSorted;
end

function [middle,spread] = medianIQR(values)
values = sort(values(isfinite(values)));
if isempty(values), middle = NaN; spread = NaN; return; end
middle = percentile(values,50);
spread = percentile(values,75)-percentile(values,25);
end

function value = percentile(sortedValues,percentage)
position = 1+(numel(sortedValues)-1)*percentage/100;
lower = floor(position); upper = ceil(position);
if lower == upper, value = sortedValues(lower);
else, value = sortedValues(lower)+(position-lower)* ...
        (sortedValues(upper)-sortedValues(lower)); end
end
