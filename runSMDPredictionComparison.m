function summary = runSMDPredictionComparison(outputRoot,numberOfRuns,includeCapped,problemNumbers)
%RUNSMDPREDICTIONCOMPARISON Paired (2+3) experiments, original vs prediction.
% Defaults: SMD1--SMD8, 10 seeds each, two variants = 160 independent runs.
% Third argument true adds a capped variant (240 runs by default).
projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_prediction_comparison', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 10; end
if nargin < 3 || isempty(includeCapped), includeCapped = false; end
if nargin < 4 || isempty(problemNumbers), problemNumbers = 1:8; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','>=',1,'<=',10});
validateattributes(includeCapped,{'logical','numeric'},{'scalar','binary'});
validateattributes(problemNumbers,{'numeric'},{'vector','integer','>=',1,'<=',8});
assert(numel(unique(problemNumbers)) == numel(problemNumbers),'Duplicate problems.');
% Refuse an existing batch directory, including a partially completed batch.
if exist(outputRoot,'dir') || exist(outputRoot,'file')
    error('BLCMAES:OutputExists','Choose a new output directory: %s',outputRoot);
end
mkdir(outputRoot);
variants = {'baseline','prediction'};
if includeCapped, variants{end+1} = 'prediction_capped'; end
% Keep the original (2+3) experiment budgets and stopping criteria.
config = struct('problemNumbers',problemNumbers,'numberOfRuns',numberOfRuns, ...
    'variants',{variants},'totalDim',5,'UmaxFEs',2500,'UmaxImprFEs',350, ...
    'LmaxFEs',250,'LmaxImprFEs',25,'weight',0.5, ...
    'startGeneration',10,'correctionInterval',1, ...
    'historyGenerations',10,'sampleCount',40, ...
    'maxCorrectionSigmaFactor',1);
save(fullfile(outputRoot,'experiment_config.mat'),'config');
% Preserve the exact source used for this batch for later reproduction.
sourceDir = fullfile(outputRoot,'source_snapshot');
mkdir(sourceDir);
sourceFiles = {'BLCMAES.m','predictionCorrectionOptions.m', ...
    'fitLocalUpperQuadratic.m','applyUpperPredictionCorrection.m', ...
    'runSMDPredictionComparison.m','testPredictionCorrection.m'};
for i = 1:numel(sourceFiles)
    copyfile(fullfile(projectRoot,sourceFiles{i}),sourceDir);
end
copyfile(fullfile(projectRoot,'BLEA_D'),fullfile(sourceDir,'BLEA_D'));
rows = cell(0,17);
columns = {'SMD','variant','run_no','seed','UF','original_F','upper_error', ...
    'upper_success','UFE','LFE_total','generations','valid_fits','corrections', ...
    'boundary_repairs','capped_steps','runtime_seconds','mat_file'};
for smdNo = problemNumbers(:)'
    baseBI = getBLOPinfo('SMD',smdNo,5);
    assert(baseBI.u_dim == 2 && baseBI.l_dim == 3);
    baseBI.UmaxFEs = config.UmaxFEs;
    baseBI.UmaxImprFEs = config.UmaxImprFEs;
    baseBI.LmaxFEs = config.LmaxFEs;
    baseBI.LmaxImprFEs = config.LmaxImprFEs;
    baseBI.u_N = 50;
    baseBI.l_N = 50;
    baseBI.generationDiagnostic = struct('enabled',true);
    for runNo = 1:numberOfRuns
        % Matches seeds in runSMDGenerationDiagnostics; identical per pair.
        seed = (smdNo-1)*10 + runNo;
        for variantIndex = 1:numel(variants)
            variant = variants{variantIndex};
            BI = baseBI;
            BI.enablePredictionCorrection = ~strcmp(variant,'baseline');
            BI.predictionCorrection = struct('logEnabled',true, ...
                'weight',config.weight, ...
                'startGeneration',config.startGeneration, ...
                'correctionInterval',config.correctionInterval, ...
                'limitCorrection', ...
                strcmp(variant,'prediction_capped'), ...
                'maxCorrectionSigmaFactor',config.maxCorrectionSigmaFactor);
            rng(seed,'twister');
            rngStateBefore = rng;
            timer = tic;
            ins = BLCMAES(BI);
            elapsed = toc(timer);
            rngStateAfter = rng;
            predictionLog = ins.predictionLog;
            metadata = struct('problem',BI.fn,'variant',variant,'runNo',runNo, ...
                'seed',seed,'rngStateBefore',rngStateBefore, ...
                'rngStateAfter',rngStateAfter,'runtimeSeconds',elapsed, ...
                'UFMeaning','UF = -original upper objective; larger is better', ...
                'generationMeaning',['Row t uses population t sampled at m_before. ' ...
                'm_new is used to sample population t+1. Terminal rows do not update.']);
            runDir = fullfile(outputRoot,sprintf('SMD%d',smdNo),'DU2_DL3',variant);
            if ~exist(runDir,'dir'), mkdir(runDir); end
            file = fullfile(runDir,sprintf('SMD%d_DU2_DL3_run%02d.mat',smdNo,runNo));
            save(file,'ins','predictionLog','BI','metadata','-v7');
            errorUF = abs(ins.UF-BI.u_fopt);
            rows(end+1,:) = {smdNo,variant,runNo,seed,ins.UF,-ins.UF,errorUF, ...
                errorUF < BI.u_ftol,ins.UFEs,ins.LFEs,numel(predictionLog), ...
                sum([predictionLog.valid_fit]),sum([predictionLog.correction_applied]), ...
                sum([predictionLog.boundary_repaired]),sum([predictionLog.cap_applied]), ...
                elapsed,file};
            % Checkpoint the manifest after each completed run.
            runs = cell2table(rows,'VariableNames',columns);
            save(fullfile(outputRoot,'comparison_runs.mat'),'runs','config');
            writetable(runs,fullfile(outputRoot,'comparison_runs.csv'));
            fprintf('SMD%d run %02d %-18s err=%.3g UFE=%d LFE=%d corrected=%d\n', ...
                smdNo,runNo,variant,errorUF,ins.UFEs,ins.LFEs, ...
                sum([predictionLog.correction_applied]));
        end
    end
end
summaryRows = cell(0,14);
for smdNo = problemNumbers(:)'
    baseline = runs(runs.SMD == smdNo & strcmp(runs.variant,'baseline'),:);
    for variantIndex = 1:numel(variants)
        variant = variants{variantIndex};
        trial = runs(runs.SMD == smdNo & strcmp(runs.variant,variant),:);
        % Rows within each variant retain the same seed/run order.
        pairedSuccess = logical(trial.upper_success) & logical(baseline.upper_success);
        successfulLFEReduction = NaN;
        if any(pairedSuccess)
            successfulLFEReduction = 100*mean(1-trial.LFE_total(pairedSuccess) ./ ...
                baseline.LFE_total(pairedSuccess));
        end
        summaryRows(end+1,:) = {smdNo,variant,height(trial), ...
            mean(trial.upper_error),median(trial.upper_error),std(trial.upper_error), ...
            sum(trial.upper_success),mean(trial.UFE),mean(trial.LFE_total), ...
            100*(1-mean(trial.UFE)/mean(baseline.UFE)), ...
            100*(1-mean(trial.LFE_total)/mean(baseline.LFE_total)), ...
            sum(pairedSuccess),successfulLFEReduction,mean(trial.corrections)};
    end
end
summary = cell2table(summaryRows,'VariableNames',{'SMD','variant','runs', ...
    'mean_upper_error','median_upper_error','std_upper_error','successes', ...
    'mean_UFE','mean_LFE','mean_UFE_reduction_pct','mean_LFE_reduction_pct', ...
    'joint_success_pairs','joint_success_mean_LFE_reduction_pct','mean_corrections'});
save(fullfile(outputRoot,'comparison_summary.mat'),'summary','runs','config');
writetable(summary,fullfile(outputRoot,'comparison_summary.csv'));
disp(summary);
fprintf('Comparison saved in %s\n',outputRoot);
end
