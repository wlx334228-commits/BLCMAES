function experiment = runSMDScaleOutcomeModelExperiment( ...
    outputRoot,numberOfRuns,problemNumbers,branchRepeats,statesPerPhase)
%RUNSMDSCALEOUTCOMEMODELEXPERIMENT Offline one-generation eta intervention.
% From an immutable pre-injection snapshot, each branch uses a test eta in
% its first resumed generation and eta=.50 for the following four.  Branches
% are evaluated after the completed main run, with private RNG streams and
% restored counters, so they cannot alter the BL-CMA-ES main trajectory.
%
% Quick correctness run:
%   runSMDScaleOutcomeModelExperiment
%
% Formal grouped-CV run (after the quick run is inspected):
%   runSMDScaleOutcomeModelExperiment([],10,[1 3 6 8],3,2)

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 1; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = [1 3]; end
if nargin < 4 || isempty(branchRepeats), branchRepeats = 1; end
if nargin < 5 || isempty(statesPerPhase), statesPerPhase = 1; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(all(ismember(problemNumbers,[1 3 6 8])), ...
    'This state-to-scale outcome experiment supports SMD1, SMD3, SMD6, and SMD8.');
validateattributes(branchRepeats,{'numeric'},{'scalar','integer','positive'});
validateattributes(statesPerPhase,{'numeric'},{'scalar','integer','positive'});

H = 5;
discreteEtas = [0.25 0.50 0.75 1.00];
% These three scales are deliberately held out from model fitting.  They
% are real branch evaluations used only to test interpolation predictions.
intermediateEtas = [0.35 0.65 0.90];
allEtas = [discreteEtas,intermediateEtas];
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_scale_outcome_model_2D', ...
        sprintf('%s_%druns_R%d_P%d_H%d',problemTag(problemNumbers), ...
        numberOfRuns,branchRepeats,statesPerPhase,H));
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end
config = makeConfig(problemNumbers,numberOfRuns,branchRepeats, ...
    statesPerPhase,H,discreteEtas,intermediateEtas,outputRoot);
saveOrValidateConfig(outputRoot,config);

snapshotRows = repmat(emptySnapshotRow(),0,1);
outcomeRows = repmat(emptyOutcomeRow(),0,1);
generationRows = repmat(emptyGenerationRow(),0,1);
branchObjects = cell(0,1);
global ulFunctionEvaluations llFunctionEvaluations;

for smdNo = problemNumbers(:)'
    for runNo = 1:numberOfRuns
        seed = mainSeed(smdNo,numberOfRuns,runNo);
        BI = makeF050BI(smdNo,seed);
        rng(seed,'twister');
        mainRngBefore = rng;
        mainTimer = tic;
        mainIns = BLCMAES(BI,struct('mode','main','captureSnapshots',true));
        mainRuntime = toc(mainTimer);
        mainRngAfter = rng;
        mainSnapshots = snapshotArray(mainIns);
        selected = selectOutcomeSnapshots(mainSnapshots,mainIns.UFEs,statesPerPhase);
        for stateIndex = 1:numel(mainSnapshots)
            snapshotRows(end+1,1) = makeSnapshotRow(mainSnapshots(stateIndex), ...
                smdNo,runNo,seed,selected(stateIndex)); %#ok<AGROW>
        end
        save(fullfile(outputRoot,sprintf('SMD%d_run%02d_main.mat',smdNo,runNo)), ...
            'mainIns','BI','mainRngBefore','mainRngAfter','mainRuntime', ...
            'mainSnapshots','selected','-v7');
        if isempty(mainSnapshots)
            fprintf('SMD%d run %d: no valid prediction snapshot.\n',smdNo,runNo);
            continue;
        end

        for stateIndex = find([selected.selected])
            snapshot = mainSnapshots(stateIndex);
            for repeatNo = 1:branchRepeats
                baseSeed = branchSeedForState(seed,snapshot,repeatNo);
                branches = repmat(struct('eta',NaN,'ins',struct(), ...
                    'info',struct(),'schedule',[]),numel(allEtas),1);
                for etaIndex = 1:numel(allEtas)
                    eta = allEtas(etaIndex);
                    schedule = [eta,repmat(0.50,1,H-1)];
                    [branches(etaIndex).ins,branches(etaIndex).info] = ...
                        runInterventionBranch(BI,snapshot,schedule,baseSeed);
                    branches(etaIndex).eta = eta;
                    branches(etaIndex).schedule = schedule;
                    generationRows = [generationRows;branchGenerationRows( ...
                        branches(etaIndex).ins,smdNo,runNo,seed,stateIndex, ...
                        selected(stateIndex),eta,repeatNo,baseSeed,schedule)]; %#ok<AGROW>
                    if config.retainBranchObjects
                        branchObjects{end+1,1} = struct('SMD',smdNo,'run_no',runNo, ...
                            'state_index',stateIndex,'repeat_no',repeatNo, ...
                            'eta',eta,'schedule',schedule, ...
                            'ins',branches(etaIndex).ins); %#ok<AGROW>
                    end
                end
                rows = outcomeRowsForMatchedBranches( ...
                    branches,snapshot,smdNo,runNo,seed,stateIndex, ...
                    selected(stateIndex),repeatNo,baseSeed,config,mainIns);
                outcomeRows = [outcomeRows;rows]; %#ok<AGROW>
            end
        end
    end
end

snapshotTable = sortableTable(snapshotRows,{'SMD','run_no','generation'});
outcomeTable = sortableTable(outcomeRows, ...
    {'SMD','run_no','state_index','repeat_no','eta'});
generationTable = sortableTable(generationRows, ...
    {'SMD','run_no','state_index','repeat_no','eta','generation'});
writetable(snapshotTable,fullfile(outputRoot,'state_snapshots.csv'));
writetable(outcomeTable,fullfile(outputRoot,'intervention_branch_outcomes.csv'));
writetable(generationTable,fullfile(outputRoot, ...
    'intervention_branch_generations.csv'));

modelResult = analyseScaleOutcomePredictionModels(outcomeTable,config);
writetable(modelResult.parameterTable,fullfile(outputRoot,'model_parameters.csv'));
writetable(modelResult.predictionTable,fullfile(outputRoot, ...
    'model_scale_predictions.csv'));
writetable(modelResult.selectionTable,fullfile(outputRoot, ...
    'model_strategy_selection.csv'));
writetable(modelResult.predictionMetrics,fullfile(outputRoot, ...
    'model_prediction_metrics.csv'));
writetable(modelResult.predictionMetricSummary,fullfile(outputRoot, ...
    'model_prediction_metric_summary.csv'));
writetable(modelResult.runStrategySummary,fullfile(outputRoot, ...
    'model_run_strategy_summary.csv'));
writetable(modelResult.strategySummary,fullfile(outputRoot, ...
    'model_strategy_summary.csv'));
writetable(modelResult.pairedDifferences,fullfile(outputRoot, ...
    'model_paired_strategy_differences.csv'));
writetable(modelResult.pairedDifferenceSummary,fullfile(outputRoot, ...
    'model_paired_strategy_summary.csv'));
writetable(modelResult.intermediateTable,fullfile(outputRoot, ...
    'intermediate_scale_validation.csv'));
figureFiles = plotScaleOutcomeModelResults(outcomeTable,modelResult, ...
    fullfile(outputRoot,'figures'));
experiment = struct('outputRoot',outputRoot,'config',config, ...
    'stateSnapshots',snapshotTable,'outcomes',outcomeTable, ...
    'generationDetails',generationTable,'modelResult',modelResult, ...
    'figureFiles',{figureFiles},'branchObjects',{branchObjects}, ...
    'notes',['All branches are offline diagnostics.  The first resumed ' ...
    'generation uses the tested eta and the remaining four use eta=.50. ' ...
    'Outcome F_end is reconstructed at a common exact UFE checkpoint.']);
save(fullfile(outputRoot,'scale_outcome_model_experiment.mat'), ...
    'experiment','-v7');
fprintf('Scale-outcome model experiment saved in %s\n',outputRoot);
end

function config = makeConfig(problems,runs,repeats,statesPerPhase,H, ...
        discreteEtas,intermediateEtas,outputRoot)
config = struct('version','scale_outcome_model_v1','totalDim',5, ...
    'upperDim',2,'lowerDim',3,'problems',problems,'numberOfRuns',runs, ...
    'branchRepeats',repeats,'statesPerPhase',statesPerPhase, ...
    'horizonGenerations',H,'interventionGenerationCount',1, ...
    'recoveryAuxiliaryScale',0.50,'discreteEtas',discreteEtas, ...
    'intermediateEtas',intermediateEtas, ...
    'retainBranchObjects',runs == 1, ...
    'composition','5 normal + 1 direct + 2 auxiliary', ...
    'predictionModel','quadratic with 40 local samples', ...
    'crossValidation','five-fold grouped by independent main run', ...
    'outcomeDefinition',['F_end at the minimum final UFE among all matched ' ...
        'eta branches; reference eta=.50']);
assert(ischar(outputRoot) || isstring(outputRoot));
end

function saveOrValidateConfig(outputRoot,config)
fileName = fullfile(outputRoot,'scale_outcome_model_config.mat');
if exist(fileName,'file') ~= 2
    save(fileName,'config');
    return;
end
loaded = load(fileName,'config');
assert(isequaln(loaded.config,config), ...
    'This output folder contains a different scale-outcome configuration.');
end

function BI = makeF050BI(smdNo,seed)
BI = getBLOPinfo('SMD',smdNo,5);
BI.UmaxFEs = 2500;
BI.UmaxImprFEs = 350;
BI.LmaxFEs = 250;
BI.LmaxImprFEs = 25;
BI.enablePredictionCorrection = false;
BI.predictionCorrection = struct('logEnabled',false);
BI.enableIndividualPredictionCorrection = false;
BI.individualPredictionCorrection = struct('mode','A0','logEnabled',false);
BI.predictionInjection = struct('mode','K3','predictionInjectCount',3, ...
    'candidatePolicy','direct_plus_auxiliary', ...
    'auxiliarySamplingDistribution','cma_gaussian', ...
    'auxiliarySamplingScale',0.50,'predictionModelType','quadratic', ...
    'logEnabled',true,'startGeneration',10,'historyGenerations',10, ...
    'useAllHistory',false,'sampleCount',40, ...
    'maxPredictionNormalizedDistance',Inf, ...
    'auxiliaryRandomSeed',500000+seed,'randomSeed',seed, ...
    'consistencyCheckEnabled',true);
end

function snapshots = snapshotArray(ins)
assert(isfield(ins,'continuousScaleSnapshots'), ...
    'BLCMAES did not return pre-injection state snapshots.');
snapshots = ins.continuousScaleSnapshots;
end

function [ins,info] = runInterventionBranch(BI,snapshot,schedule,baseSeed)
global ulFunctionEvaluations llFunctionEvaluations;
savedRng = rng;
savedUFE = ulFunctionEvaluations;
savedLFE = llFunctionEvaluations;
cleanup = onCleanup(@() restoreOuterState(savedRng,savedUFE,savedLFE)); %#ok<NASGU>
resumeState = snapshot;
resumeState.injection_state.auxiliaryStream = branchStreamDescription(baseSeed,71);
control = struct('mode','branch','resumeState',resumeState, ...
    'fixedAuxiliaryScale',schedule(1), ...
    'auxiliaryScaleSchedule',schedule, ...
    'horizonGenerations',numel(schedule),'branchRandomSeed',baseSeed, ...
    'forceFixedHorizon',true);
timer = tic;
ins = BLCMAES(BI,control);
info = struct('runtimeSeconds',toc(timer),'UFE',ins.UFEs,'LFE',ins.LFEs);
end

function rows = outcomeRowsForMatchedBranches( ...
        branches,snapshot,smdNo,runNo,seed,stateIndex,selection,repeatNo, ...
        baseSeed,config,mainIns)
finalUFEs = arrayfun(@(b)b.ins.UFEs,branches);
commonUFE = min(finalUFEs);
assert(commonUFE >= snapshot.UFE_before_sampling, ...
    'A branch ended before its resume snapshot UFE.');
Fend = NaN(numel(branches),1);
for index = 1:numel(branches)
    Fend(index) = historyBestAtUFE(snapshot,branches(index).ins,commonUFE);
end
referenceIndex = find(abs([branches.eta]-0.50) < 10*eps,1,'first');
assert(~isempty(referenceIndex) && isfinite(Fend(referenceIndex)), ...
    'Matched eta=.50 branch is missing or has no aligned historical best.');
referenceF = Fend(referenceIndex);
scaleDenominator = snapshot.sF + eps(max(1,snapshot.sF));
rows = repmat(emptyOutcomeRow(),0,1);
for index = 1:numel(branches)
    branch = branches(index);
    row = emptyOutcomeRow();
    row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
    row.run_key = sprintf('SMD%d_run%03d',smdNo,runNo);
    row.state_index = stateIndex; row.state_id = selection.state_id;
    row.state_key = sprintf('SMD%d_run%03d_state%03d_rep%03d', ...
        smdNo,runNo,stateIndex,repeatNo);
    row.phase = selection.phase; row.phase_progress = selection.progress;
    row.repeat_no = repeatNo; row.branch_seed = baseSeed;
    row.eta = branch.eta; row.eta_label = etaLabel(branch.eta);
    row.is_training_eta = double(any(abs(branch.eta-config.discreteEtas) < 10*eps));
    row.is_intermediate_eta = double(any(abs(branch.eta-config.intermediateEtas) < 10*eps));
    row.is_reference_eta = double(abs(branch.eta-0.50) < 10*eps);
    row.auxiliary_scale_schedule = vectorText(branch.schedule);
    row.start_generation = snapshot.generation;
    row.completed_generations = numel(branch.ins.predictionInjectionLog);
    row.d = snapshot.d; row.q = snapshot.q; row.sF = snapshot.sF;
    row.sigma_start = snapshot.sigma;
    row.history_best_start_UF = snapshot.history_best_UF;
    row.common_UFE_checkpoint = commonUFE;
    row.common_additional_UFE = commonUFE-snapshot.UFE_before_sampling;
    row.branch_UFE_end = branch.ins.UFEs;
    row.branch_LFE_end = branch.ins.LFEs;
    row.branch_UFE_delta = branch.ins.UFEs-snapshot.UFE_before_sampling;
    row.branch_LFE_delta = branch.ins.LFEs-snapshot.LFE_before_sampling;
    row.F_end = Fend(index); row.F_end_reference_eta050 = referenceF;
    row.relative_overall_gain = (Fend(index)-referenceF)/scaleDenominator;
    % Use the real SMD optimum stored in the original main problem.
    row.final_upper_error = abs(branch.ins.UF-snapshot.BI.u_fopt);
    row.final_lower_error = abs(branch.ins.LF-snapshot.BI.l_fopt);
    row.main_UFE_end = mainIns.UFEs; row.main_LFE_end = mainIns.LFEs;
    row.valid_prediction_generations = sum([branch.ins.predictionInjectionLog.prediction_valid_used]);
    row.prediction_fallback_generations = row.completed_generations - ...
        row.valid_prediction_generations;
    row.branch_runtime_seconds = branch.info.runtimeSeconds;
    rows(end+1,1) = row; %#ok<AGROW>
end
end

function best = historyBestAtUFE(snapshot,ins,targetUFE)
% Reconstruct only evaluations actually completed on or before targetUFE.
% Candidate counts come from the evaluation loop; optional refinement counts
% are logged separately.  This avoids comparing endpoints that consumed
% different extra UFE merely because one branch refined more often.
best = snapshot.history_best_UF;
if ~isfinite(best), best = -Inf; end
log = ins.predictionInjectionLog;
for generationIndex = 1:numel(log)
    entry = log(generationIndex);
    counts = entry.candidate_upper_evaluation_UFE(:);
    values = entry.upper_objective_after_true_evaluation(:);
    [counts,order] = sort(counts);
    values = values(order);
    for item = 1:numel(counts)
        if isfinite(counts(item)) && counts(item) <= targetUFE && ...
                isfinite(values(item))
            best = max(best,values(item));
        end
    end
    best = includeRefinement(best,entry,'population_refinement',targetUFE);
    best = includeRefinement(best,entry,'external_refinement',targetUFE);
end
end

function best = includeRefinement(best,entry,prefix,targetUFE)
flag = [prefix '_upper_evaluated'];
ufName = [prefix '_UF'];
ufeName = [prefix '_UFE'];
if isfield(entry,flag) && logical(entry.(flag)) && isfield(entry,ufName) && ...
        isfield(entry,ufeName) && isfinite(entry.(ufeName)) && ...
        entry.(ufeName) <= targetUFE && isfinite(entry.(ufName))
    best = max(best,entry.(ufName));
end
end

function rows = branchGenerationRows(ins,smdNo,runNo,seed,stateIndex, ...
        selection,eta,repeatNo,branchSeed,schedule)
log = ins.predictionInjectionLog;
rows = repmat(emptyGenerationRow(),0,1);
for index = 1:numel(log)
    entry = log(index);
    row = emptyGenerationRow();
    row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
    row.state_index = stateIndex; row.state_id = selection.state_id;
    row.phase = selection.phase; row.repeat_no = repeatNo;
    row.branch_seed = branchSeed; row.eta = eta; row.eta_label = etaLabel(eta);
    row.generation = entry.generation; row.generation_offset = index-1;
    row.scheduled_eta = schedule(index);
    row.used_eta = entry.auxiliary_sampling_scale;
    row.prediction_valid = double(entry.prediction_valid_used);
    row.injection_reason = entry.injection_reason;
    row.normal_count = sum(strcmp(entry.source_labels,'normal'));
    row.direct_count = sum(strcmp(entry.source_labels,'predicted_direct'));
    row.auxiliary_count = sum(strcmp(entry.source_labels,'predicted_auxiliary'));
    row.UFE_end = entry.UFE; row.LFE_end = entry.LFE_total;
    row.best_so_far_UF = entry.best_so_far_UF;
    rows(end+1,1) = row; %#ok<AGROW>
end
end

function rows = selectOutcomeSnapshots(snapshots,terminalUFE,statesPerPhase)
rows = repmat(struct('selected',false,'selection_rank',NaN, ...
    'state_index',NaN,'state_id','','phase','','progress',NaN, ...
    'reason',''),numel(snapshots),1);
if isempty(snapshots), return; end
startUFE = snapshots(1).UFE_before_sampling;
denominator = terminalUFE-startUFE;
progress = NaN(numel(snapshots),1);
if denominator > 0
    progress = ([snapshots.UFE_before_sampling]'-startUFE)/denominator;
end
phaseNames = {'early','middle','late'};
phaseTargets = [1/6,1/2,5/6];
for index = 1:numel(snapshots)
    rows(index).state_index = index;
    rows(index).state_id = sprintf('gen%04d',snapshots(index).generation);
    rows(index).phase = phaseForProgress(progress(index));
    rows(index).progress = progress(index);
    rows(index).reason = 'not_selected';
end
allD = [snapshots.d]'; allQ = [snapshots.q]';
dScale = max(std(allD,0,'omitnan'),eps);
qScale = max(std(allQ,0,'omitnan'),eps);
for phaseIndex = 1:numel(phaseNames)
    candidates = find(strcmp({rows.phase},phaseNames{phaseIndex}));
    chosen = zeros(0,1);
    for rankIndex = 1:min(statesPerPhase,numel(candidates))
        if isempty(chosen)
            score = abs(progress(candidates)-phaseTargets(phaseIndex));
            reason = 'phase_midpoint';
        else
            score = zeros(numel(candidates),1);
            for localIndex = 1:numel(candidates)
                candidate = candidates(localIndex);
                if any(chosen == candidate)
                    score(localIndex) = Inf;
                    continue;
                end
                dDistance = abs(snapshots(candidate).d-[snapshots(chosen).d])/dScale;
                qDistance = abs(snapshots(candidate).q-[snapshots(chosen).q])/qScale;
                novelty = min(sqrt(dDistance.^2+qDistance.^2));
                % Diversity is primary after the midpoint state; the tiny
                % progress term prevents arbitrary ties far outside a phase.
                score(localIndex) = -novelty + 1e-3* ...
                    abs(progress(candidate)-phaseTargets(phaseIndex));
            end
            reason = 'phase_diverse_dq';
        end
        [~,localBest] = min(score);
        best = candidates(localBest);
        chosen(end+1,1) = best; %#ok<AGROW>
        rows(best).selected = true;
        rows(best).selection_rank = rankIndex;
        rows(best).reason = reason;
    end
end
end

function phase = phaseForProgress(progress)
if ~isfinite(progress), phase = 'unclassified';
elseif progress < 1/3, phase = 'early';
elseif progress < 2/3, phase = 'middle';
else, phase = 'late';
end
end

function row = makeSnapshotRow(snapshot,smdNo,runNo,seed,selection)
row = emptySnapshotRow();
row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
row.state_index = selection.state_index; row.state_id = selection.state_id;
row.selected = double(selection.selected); row.selection_rank = selection.selection_rank;
row.phase = selection.phase; row.phase_progress = selection.progress;
row.selection_reason = selection.reason; row.generation = snapshot.generation;
row.UFE_before_sampling = snapshot.UFE_before_sampling;
row.LFE_before_sampling = snapshot.LFE_before_sampling;
row.d = snapshot.d; row.q = snapshot.q; row.sigma = snapshot.sigma;
row.sF = snapshot.sF; row.history_best_UF = snapshot.history_best_UF;
row.x_pred = vectorText(snapshot.x_pred_actual);
row.cma_mean = vectorText(snapshot.cma_mean(1:2));
row.Cuu = matrixText(snapshot.Cuu);
row.H_physical = matrixText(snapshot.H_physical);
end

function tableData = sortableTable(rows,sortNames)
if isempty(rows)
    tableData = struct2table(rows);
else
    tableData = sortrows(struct2table(rows),sortNames);
end
end

function description = branchStreamDescription(baseSeed,streamKind)
streamSeed = mod(double(baseSeed)+1009*double(streamKind),2^31-2)+1;
stream = RandStream('mt19937ar','Seed',floor(streamSeed));
description = struct('present',true,'type',stream.Type,'seed',stream.Seed, ...
    'state',stream.State);
end

function restoreOuterState(savedRng,savedUFE,savedLFE)
global ulFunctionEvaluations llFunctionEvaluations;
rng(savedRng);
ulFunctionEvaluations = savedUFE;
llFunctionEvaluations = savedLFE;
end

function seed = mainSeed(smdNo,runs,runNo)
seed = (smdNo-1)*runs + runNo;
end

function seed = branchSeedForState(runSeed,snapshot,repeatNo)
seed = mod(double(runSeed)+1000003*double(snapshot.generation)+ ...
    8191*double(repeatNo),2^31-2)+1;
seed = floor(seed);
end

function label = etaLabel(eta)
label = sprintf('eta_%03d',round(100*eta));
end

function text = vectorText(values)
text = mat2str(values,16);
end

function text = matrixText(values)
text = mat2str(values,16);
end

function tag = problemTag(problems)
tag = sprintf('SMD%d_',problems); tag(end) = [];
end

function row = emptySnapshotRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_index',NaN, ...
    'state_id','','selected',NaN,'selection_rank',NaN,'phase','', ...
    'phase_progress',NaN,'selection_reason','','generation',NaN, ...
    'UFE_before_sampling',NaN,'LFE_before_sampling',NaN,'d',NaN,'q',NaN, ...
    'sigma',NaN,'sF',NaN,'history_best_UF',NaN,'x_pred','', ...
    'cma_mean','','Cuu','','H_physical','');
end

function row = emptyOutcomeRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'run_key','', ...
    'state_index',NaN,'state_id','','state_key','','phase','', ...
    'phase_progress',NaN,'repeat_no',NaN,'branch_seed',NaN, ...
    'eta',NaN,'eta_label','','is_training_eta',NaN, ...
    'is_intermediate_eta',NaN,'is_reference_eta',NaN, ...
    'auxiliary_scale_schedule','','start_generation',NaN, ...
    'completed_generations',NaN,'d',NaN,'q',NaN,'sF',NaN, ...
    'sigma_start',NaN,'history_best_start_UF',NaN, ...
    'common_UFE_checkpoint',NaN,'common_additional_UFE',NaN, ...
    'branch_UFE_end',NaN,'branch_LFE_end',NaN, ...
    'branch_UFE_delta',NaN,'branch_LFE_delta',NaN, ...
    'F_end',NaN,'F_end_reference_eta050',NaN, ...
    'relative_overall_gain',NaN,'final_upper_error',NaN, ...
    'final_lower_error',NaN,'main_UFE_end',NaN,'main_LFE_end',NaN, ...
    'valid_prediction_generations',NaN,'prediction_fallback_generations',NaN, ...
    'branch_runtime_seconds',NaN);
end

function row = emptyGenerationRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_index',NaN, ...
    'state_id','','phase','','repeat_no',NaN,'branch_seed',NaN, ...
    'eta',NaN,'eta_label','','generation',NaN,'generation_offset',NaN, ...
    'scheduled_eta',NaN,'used_eta',NaN,'prediction_valid',NaN, ...
    'injection_reason','','normal_count',NaN,'direct_count',NaN, ...
    'auxiliary_count',NaN,'UFE_end',NaN,'LFE_end',NaN,'best_so_far_UF',NaN);
end
