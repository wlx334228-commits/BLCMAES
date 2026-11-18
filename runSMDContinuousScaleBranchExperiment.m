function experiment = runSMDContinuousScaleBranchExperiment( ...
    outputRoot,numberOfRuns,problemNumbers,horizonGenerations,branchRepeats)
%RUNSMDCONTINUOUSSCALEBRANCHEXPERIMENT Stateful multi-generation eta test.
% A completed F050 main run is never altered.  At every valid pending
% quadratic prediction BLCMAES stores a read-only pre-injection snapshot.
% After the main run finishes, this runner selects early/middle/late states
% and resumes isolated branches for a fixed H-generation horizon at
% eta={.25,.50,.75,1.0}, plus an A1 no-auxiliary control branch.
%
% Small first-round experiment requested for this study:
%   runSMDContinuousScaleBranchExperiment
%
% A longer H=10 confirmation can be run in a different result folder:
%   runSMDContinuousScaleBranchExperiment([],3,[1 3],10,3)

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 3; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = [1 3]; end
if nargin < 4 || isempty(horizonGenerations), horizonGenerations = 5; end
if nargin < 5 || isempty(branchRepeats), branchRepeats = 1; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',8});
assert(all(ismember(problemNumbers,[1 3])), ...
    'The initial state-branch study is limited to SMD1 and SMD3.');
validateattributes(horizonGenerations,{'numeric'}, ...
    {'scalar','integer','positive'});
validateattributes(branchRepeats,{'numeric'}, ...
    {'scalar','integer','positive'});
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot, ...
        'results_continuous_scale_branch_2D',sprintf( ...
        '%s_%druns_H%d_R%d',problemTag(problemNumbers),numberOfRuns, ...
        horizonGenerations,branchRepeats));
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end
config = makeConfig(problemNumbers,numberOfRuns,horizonGenerations, ...
    branchRepeats,outputRoot);
saveOrValidateConfig(outputRoot,config);

snapshotRows = repmat(emptySnapshotRow(),0,1);
generationRows = repmat(emptyGenerationRow(),0,1);
auxiliaryRows = repmat(emptyAuxiliaryRow(),0,1);
summaryRows = repmat(emptyBranchSummaryRow(),0,1);
branchObjects = cell(0,1);
global ulFunctionEvaluations llFunctionEvaluations;

for smdNo = problemNumbers(:)'
    for runNo = 1:numberOfRuns
        seed = (smdNo-1)*numberOfRuns + runNo;
        BI = makeF050BI(smdNo,seed);
        rng(seed,'twister');
        mainRngBefore = rng;
        mainTimer = tic;
        mainIns = BLCMAES(BI,struct('mode','main','captureSnapshots',true));
        mainRuntime = toc(mainTimer);
        mainRngAfter = rng;
        mainSnapshots = snapshotArray(mainIns);
        selected = selectContinuousScaleSnapshots(mainSnapshots,mainIns.UFEs);
        if isempty(mainSnapshots)
            fprintf(['SMD%d run %d: no valid quadratic-prediction state; ' ...
                'main result saved and no branches started.\n'],smdNo,runNo);
        end
        for stateIndex = 1:numel(mainSnapshots)
            snapshotRows(end+1,1) = makeSnapshotRow(mainSnapshots(stateIndex), ...
                smdNo,runNo,seed,selected(stateIndex)); %#ok<AGROW>
        end
        mainFile = fullfile(outputRoot,sprintf('SMD%d_run%02d_main.mat', ...
            smdNo,runNo));
        save(mainFile,'mainIns','BI','mainRngBefore','mainRngAfter', ...
            'mainRuntime','mainSnapshots','selected','-v7');

        for stateIndex = find([selected.selected])
            snapshot = mainSnapshots(stateIndex);
            for repeatNo = 1:branchRepeats
                sharedBranchSeed = branchSeedForState(seed,snapshot,repeatNo);
                for eta = config.etas
                    [branchIns,branchInfo] = runOneBranch(BI,snapshot,eta, ...
                        horizonGenerations,sharedBranchSeed);
                    [details,auxRows,summary] = collectBranchEvidence( ...
                        branchIns,snapshot,smdNo,runNo,seed,stateIndex, ...
                        selected(stateIndex),eta,repeatNo,sharedBranchSeed,BI);
                    generationRows = [generationRows;details]; %#ok<AGROW>
                    auxiliaryRows = [auxiliaryRows;auxRows]; %#ok<AGROW>
                    summary.branch_runtime_seconds = branchInfo.runtimeSeconds;
                    summary.main_UFE_end = mainIns.UFEs;
                    summary.main_LFE_end = mainIns.LFEs;
                    summaryRows(end+1,1) = summary; %#ok<AGROW>
                    branchObjects{end+1,1} = struct('SMD',smdNo, ...
                        'run_no',runNo,'state_index',stateIndex, ...
                        'eta',eta,'repeat_no',repeatNo,'ins',branchIns); %#ok<AGROW>
                end
                % A1 is a matched stateful control: it runs H generations
                % with the same direct prediction but no auxiliary members.
                [a1Ins,a1Info] = runOneBranch(makeA1BI(BI),snapshot,NaN, ...
                    horizonGenerations,sharedBranchSeed);
                [details,auxRows,summary] = collectBranchEvidence( ...
                    a1Ins,snapshot,smdNo,runNo,seed,stateIndex, ...
                    selected(stateIndex),NaN,repeatNo,sharedBranchSeed, ...
                    makeA1BI(BI));
                generationRows = [generationRows;details]; %#ok<AGROW>
                auxiliaryRows = [auxiliaryRows;auxRows]; %#ok<AGROW>
                summary.branch_runtime_seconds = a1Info.runtimeSeconds;
                summary.main_UFE_end = mainIns.UFEs;
                summary.main_LFE_end = mainIns.LFEs;
                summaryRows(end+1,1) = summary; %#ok<AGROW>
                branchObjects{end+1,1} = struct('SMD',smdNo, ...
                    'run_no',runNo,'state_index',stateIndex, ...
                    'eta',NaN,'repeat_no',repeatNo,'ins',a1Ins); %#ok<AGROW>
            end
        end
    end
end

snapshotTable = sortrows(struct2table(snapshotRows), ...
    {'SMD','run_no','generation'});
generationTable = sortrows(struct2table(generationRows), ...
    {'SMD','run_no','state_index','eta_label','repeat_no','generation','candidate_index'});
auxiliaryTable = sortrows(struct2table(auxiliaryRows), ...
    {'SMD','run_no','state_index','eta_label','repeat_no','generation','candidate_index'});
branchSummary = sortrows(struct2table(summaryRows), ...
    {'SMD','run_no','state_index','eta_label','repeat_no'});
writetable(snapshotTable,fullfile(outputRoot,'state_snapshots.csv'));
writetable(generationTable,fullfile(outputRoot,'branch_generation_details.csv'));
writetable(auxiliaryTable,fullfile(outputRoot,'auxiliary_candidate_details.csv'));
writetable(branchSummary,fullfile(outputRoot,'branch_scale_summary.csv'));
[branchVsA1] = compareContinuousScaleBranchesToA1(branchSummary);
writetable(branchVsA1,fullfile(outputRoot,'branch_vs_A1_summary.csv'));
[stateScaleAnalysis,regressionTable] = analyseContinuousScaleBranches( ...
    branchSummary,snapshotTable);
writetable(stateScaleAnalysis,fullfile(outputRoot,'state_scale_analysis.csv'));
writetable(regressionTable,fullfile(outputRoot,'state_scale_regression.csv'));
figureFiles = plotContinuousScaleBranchResults(branchSummary,snapshotTable, ...
    fullfile(outputRoot,'figures'));
writeContinuousScaleBranchHtmlReport( ...
    fullfile(outputRoot,'continuous_scale_branch_report.html'), ...
    config,branchSummary,stateScaleAnalysis,figureFiles);
experiment = struct('outputRoot',outputRoot,'config',config, ...
    'stateSnapshots',snapshotTable,'generationDetails',generationTable, ...
    'auxiliaryCandidates',auxiliaryTable,'branchSummary',branchSummary, ...
    'branchVsA1',branchVsA1, ...
    'stateScaleAnalysis',stateScaleAnalysis,'regression',regressionTable, ...
    'figureFiles',{figureFiles},'branchObjects',{branchObjects}, ...
    'notes',['Branches are post-run diagnostics. Their UFE/LFE are saved ' ...
        'separately and never enter main-run counters or CMA-ES state.']);
save(fullfile(outputRoot,'continuous_scale_branch_experiment.mat'), ...
    'experiment','-v7');
fprintf('Continuous multi-generation scale experiment saved in %s\n',outputRoot);
end

function config = makeConfig(problems,runs,H,repeats,outputRoot)
config = struct('version','continuous_scale_branch_v1','totalDim',5, ...
    'upperDim',2,'lowerDim',3,'problems',problems,'numberOfRuns',runs, ...
    'horizonGenerations',H,'branchRepeats',repeats, ...
    'etas',[0.25 0.50 0.75 1.00],'mainAuxiliaryScale',0.50, ...
    'mainComposition','5 normal + 1 direct + 2 auxiliary', ...
    'controlComposition','7 normal + 1 direct (A1)', ...
    'predictionModel','quadratic','sampleCount',40, ...
    'snapshotTiming','after normal sampling, before prediction injection');
% Retain the argument in this signature for a stable public runner call;
% never persist an absolute path so a completed experiment remains movable.
assert(ischar(outputRoot) || isstring(outputRoot));
end

function saveOrValidateConfig(outputRoot,config)
fileName = fullfile(outputRoot,'continuous_scale_branch_config.mat');
if exist(fileName,'file') ~= 2
    save(fileName,'config');
    return;
end
loaded = load(fileName,'config');
assert(isequaln(loaded.config,config), ...
    'This output folder contains a different continuous-branch configuration.');
end

function BI = makeF050BI(smdNo,seed)
BI = getBLOPinfo('SMD',smdNo,5);
% getBLOPinfo defines the SMD geometry and tolerance only.  The original
% BL-CMA-ES evaluation budgets are assigned by every existing experiment
% runner, so the standalone continuous-branch runner must do the same.
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

function BI = makeA1BI(BI)
BI.predictionInjection.mode = 'K1';
BI.predictionInjection.predictionInjectCount = 1;
% K1 has no auxiliary stream and therefore retains 7 normal + 1 direct.
end

function snapshots = snapshotArray(ins)
assert(isfield(ins,'continuousScaleSnapshots'), ...
    'BLCMAES did not return continuous-scale snapshots.');
snapshots = ins.continuousScaleSnapshots;
end

function [ins,info] = runOneBranch(BI,snapshot,eta,H,baseSeed)
global ulFunctionEvaluations llFunctionEvaluations;
savedRng = rng;
savedUFE = ulFunctionEvaluations;
savedLFE = llFunctionEvaluations;
cleanup = onCleanup(@() restoreDiagnosticOuterState(savedRng,savedUFE,savedLFE)); %#ok<NASGU>
resumeState = snapshot;
% Each repeat receives a fresh private auxiliary stream, while all eta
% branches within that repeat receive its identical state.  This preserves
% paired directions and avoids treating repeated copies of one draw as
% independent evidence.
resumeState.injection_state.auxiliaryStream = branchStreamDescription( ...
    baseSeed,71);
if BI.predictionInjection.predictionInjectCount == 1
    % K3 repairs an out-of-bound direct point against raw slot 6; K1 would
    % normally use slot 8.  The direct member replaces that K1 slot anyway,
    % so copying this unused repair reference makes the A1 control's upper
    % prediction exactly match the eta branches even in the rare repaired
    % prediction case.  Its seven normal members (slots 1:7) stay intact.
    k3DirectIndex = size(resumeState.raw_normal_upper,1) - 3 + 1;
    k1DirectIndex = size(resumeState.raw_normal_upper,1);
    resumeState.raw_normal_upper(k1DirectIndex,:) = ...
        resumeState.raw_normal_upper(k3DirectIndex,:);
end
control = struct('mode','branch','resumeState',resumeState, ...
    'fixedAuxiliaryScale',eta,'horizonGenerations',H, ...
    'branchRandomSeed',baseSeed,'forceFixedHorizon',true);
if isnan(eta)
    % A1 never calls the auxiliary sampler.  Its scale is deliberately a
    % harmless finite value only to satisfy the branch-control validator.
    control.fixedAuxiliaryScale = 0.5;
end
timer = tic;
ins = BLCMAES(BI,control);
info = struct('runtimeSeconds',toc(timer),'UFE',ins.UFEs, ...
    'LFE',ins.LFEs);
end

function description = branchStreamDescription(baseSeed,streamKind)
streamSeed = mod(double(baseSeed)+1009*double(streamKind),2^31-2)+1;
stream = RandStream('mt19937ar','Seed',floor(streamSeed));
description = struct('present',true,'type',stream.Type,'seed',stream.Seed, ...
    'state',stream.State);
end

function restoreDiagnosticOuterState(savedRng,savedUFE,savedLFE)
global ulFunctionEvaluations llFunctionEvaluations;
rng(savedRng);
ulFunctionEvaluations = savedUFE;
llFunctionEvaluations = savedLFE;
end

function seed = branchSeedForState(runSeed,snapshot,repeatNo)
seed = mod(double(runSeed) + 1000003*double(snapshot.generation) + ...
    8191*double(repeatNo),2^31-2) + 1;
seed = floor(seed);
end

function row = makeSnapshotRow(snapshot,smdNo,runNo,seed,selection)
row = emptySnapshotRow();
row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
row.state_index = selection.state_index; row.state_id = selection.state_id;
row.selected = double(selection.selected); row.phase = selection.phase;
row.phase_progress = selection.progress; row.selection_reason = selection.reason;
row.generation = snapshot.generation;
row.UFE_before_sampling = snapshot.UFE_before_sampling;
row.LFE_before_sampling = snapshot.LFE_before_sampling;
row.d = snapshot.d; row.q = snapshot.q; row.sigma = snapshot.sigma;
row.sF = snapshot.sF; row.model_scale = snapshot.model_scale;
row.model_design_condition = snapshot.model_design_condition;
row.model_hessian_rcond = snapshot.model_hessian_rcond;
row.history_best_UF = snapshot.history_best_UF;
row.x_pred = vectorText(snapshot.x_pred_actual);
row.cma_mean = vectorText(snapshot.cma_mean(1:2));
row.Cuu = matrixText(snapshot.Cuu);
row.H_normalized = matrixText(snapshot.H_normalized);
row.H_physical = matrixText(snapshot.H_physical);
row.raw_normal_joint = matrixText(snapshot.normal_joint);
end

function [detailRows,auxRows,summary] = collectBranchEvidence( ...
        ins,snapshot,smdNo,runNo,seed,stateIndex,selection,eta,repeatNo, ...
        branchSeed,BI)
log = ins.predictionInjectionLog;
detailRows = repmat(emptyGenerationRow(),0,1);
auxRows = repmat(emptyAuxiliaryRow(),0,1);
baseBest = snapshot.history_best_UF;
if ~isfinite(baseBest), baseBest = -Inf; end
allAuxUF = []; allD = []; allPairWins = []; allPairTotal = 0;
auxWinGeneration = []; auxiliaryParents = 0; auxiliaryCount = 0;
auxRefreshCount = 0; auxRefreshAmount = [];
endBest = baseBest;
directDifferences = [];
for generationIndex = 1:numel(log)
    entry = log(generationIndex);
    labels = entry.source_labels;
    % The pre-refinement column is the one true evaluation of every sampled
    % candidate at its generated position.  A winning candidate can receive
    % the pre-existing optional refinement before CMA-ES updates, so retain
    % that second value separately rather than conflating candidate quality
    % with the update-state quality.
    UF = entry.upper_objective_after_true_evaluation(:);
    UFForUpdate = entry.population_UF(:);
    fitRanks = entry.fitness_rank(:);
    normalIndices = find(strcmp(labels,'normal'));
    directIndices = find(strcmp(labels,'predicted_direct'));
    auxiliaryIndices = find(strcmp(labels,'predicted_auxiliary'));
    normalBest = NaN;
    if ~isempty(normalIndices), normalBest = max(UF(normalIndices)); end
    directUF = NaN;
    if ~isempty(directIndices), directUF = UF(directIndices(1)); end
    selected = entry.selected_parent_indices(:)';
    weights = cmaWeights(numel(labels));
    runningBest = endBest;
    generationAuxBest = -Inf;
    for candidateIndex = 1:numel(labels)
        source = labels{candidateIndex};
        parentPosition = find(selected == candidateIndex,1,'first');
        parentSelected = ~isempty(parentPosition);
        parentWeight = NaN;
        if parentSelected, parentWeight = weights(parentPosition); end
        trueJoint = entry.joint_after_true_evaluation(candidateIndex,:);
        updateJoint = entry.joint_for_cma_update(candidateIndex,:);
        row = emptyGenerationRow();
        row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
        row.state_index = stateIndex; row.state_id = selection.state_id;
        row.phase = selection.phase; row.eta = eta; row.eta_label = etaLabel(eta);
        row.repeat_no = repeatNo; row.branch_seed = branchSeed;
        row.generation = entry.generation; row.generation_offset = generationIndex-1;
        row.source = source; row.candidate_index = candidateIndex;
        row.UF = UF(candidateIndex);
        row.UF_for_cma_update = UFForUpdate(candidateIndex);
        row.fitness_rank = fitRanks(candidateIndex);
        row.parent_selected = double(parentSelected); row.parent_weight = parentWeight;
        row.normal_best_UF = normalBest; row.direct_UF = directUF;
        row.prediction_point_upper = vectorText(entry.x_pred_actual);
        row.prediction_point_UF = directUF;
        row.joint = vectorText(trueJoint);
        row.joint_for_cma_update = vectorText(updateJoint);
        row.cma_mean_before = ...
            vectorText(entry.cma_mean_before_sampling(1:BI.u_dim));
        row.pc_before = vectorText(entry.pc_before_sampling);
        row.ps_before = vectorText(entry.ps_before_sampling);
        row.sigma_before = entry.sigma_before_sampling;
        row.Cuu_before = matrixText(entry.C_before_sampling(1:BI.u_dim,1:BI.u_dim));
        row.cma_mean_after = vectorText(entry.cma_mean_after_update(1:BI.u_dim));
        row.sigma_after = entry.sigma_after_update;
        row.C_after = matrixText(entry.C_after_update);
        row.prediction_valid = double(entry.prediction_valid_used);
        row.injection_reason = entry.injection_reason;
        row.UFE = entry.UFE; row.LFE_total = entry.LFE_total;
        detailRows(end+1,1) = row; %#ok<AGROW>
        if strcmp(source,'predicted_auxiliary')
            aux = emptyAuxiliaryRow();
            aux = copyCommonAuxiliaryFields(aux,row);
            aux.raw_joint = auxiliaryJointText(entry,candidateIndex,'raw');
            aux.actual_joint = vectorText(trueJoint);
            aux.actual_joint_for_cma_update = vectorText(updateJoint);
            aux.raw_distance_to_prediction = ...
                auxiliaryDistance(entry,candidateIndex,'raw',BI.u_dim);
            aux.distance_to_prediction = rowDistance(entry,candidateIndex,BI.u_dim);
            aux.relative_distance_to_cma_scale = relativeDistance(entry,candidateIndex);
            aux.auxiliary_sampling_eta = entry.auxiliary_sampling_scale;
            aux.auxiliary_sigma = entry.auxiliary_sampling_scale * ...
                entry.sigma_before_sampling;
            aux.auxiliary_rms_search_scale = ...
                entry.auxiliary_sampling_scale * ...
                entry.cma_upper_rms_search_scale;
            aux.boundary_repaired = double(candidateBoundaryRepair(entry,candidateIndex));
            aux.better_than_direct = double(isfinite(directUF) && UF(candidateIndex) > directUF);
            aux.better_than_normal_best = double(isfinite(normalBest) && UF(candidateIndex) > normalBest);
            aux.pair_wins = sum(UF(candidateIndex) > UF(normalIndices));
            aux.pair_total = numel(normalIndices);
            aux.history_best_before_candidate = runningBest;
            aux.history_refresh = double(UF(candidateIndex) > runningBest);
            aux.history_improvement = max(0,UF(candidateIndex)-runningBest);
            auxRows(end+1,1) = aux; %#ok<AGROW>
            if aux.history_refresh
                auxRefreshCount = auxRefreshCount + 1;
                auxRefreshAmount(end+1,1) = aux.history_improvement; %#ok<AGROW>
            end
            generationAuxBest = max(generationAuxBest,UF(candidateIndex));
            allAuxUF(end+1,1) = UF(candidateIndex); %#ok<AGROW>
            allPairWins(end+1,1) = aux.pair_wins; %#ok<AGROW>
            allPairTotal = allPairTotal + aux.pair_total;
            auxiliaryParents = auxiliaryParents + parentSelected;
            auxiliaryCount = auxiliaryCount + 1;
            if isfinite(directUF), directDifferences(end+1,1) = UF(candidateIndex)-directUF; end %#ok<AGROW>
        end
        runningBest = max(runningBest,UF(candidateIndex));
    end
    endBest = max(endBest,max(UF));
    if isfinite(generationAuxBest)
        allD(end+1,1) = generationAuxBest-normalBest; %#ok<AGROW>
        auxWinGeneration(end+1,1) = generationAuxBest > normalBest; %#ok<AGROW>
    end
    if isfinite(entry.best_so_far_UF), endBest = max(endBest,entry.best_so_far_UF); end
end
summary = emptyBranchSummaryRow();
summary.SMD = smdNo; summary.run_no = runNo; summary.seed = seed;
summary.state_index = stateIndex; summary.state_id = selection.state_id;
summary.phase = selection.phase; summary.eta = eta; summary.eta_label = etaLabel(eta);
summary.repeat_no = repeatNo; summary.branch_seed = branchSeed;
summary.start_generation = snapshot.generation;
summary.requested_horizon_generations = ins.continuousScaleBranch.requestedGenerations;
summary.completed_generations = numel(log);
summary.early_termination = double(summary.completed_generations < ...
    summary.requested_horizon_generations);
summary.valid_prediction_generations = sum([log.prediction_valid_used]);
summary.prediction_fallback_generations = numel(log)-summary.valid_prediction_generations;
summary.d = snapshot.d; summary.q = snapshot.q; summary.sigma_start = snapshot.sigma;
summary.sF = snapshot.sF; summary.history_best_start_UF = baseBest;
summary.history_best_end_UF = endBest;
summary.overall_history_improvement = max(0,endBest-baseBest);
summary.auxiliary_best_UF = finiteMax(allAuxUF);
summary.auxiliary_improvement_over_start = max(0,summary.auxiliary_best_UF-baseBest);
summary.auxiliary_history_refresh_count = auxRefreshCount;
summary.auxiliary_history_refresh_rate = safeRatio(auxRefreshCount,auxiliaryCount);
summary.auxiliary_refresh_improvement_total = sum(auxRefreshAmount);
summary.auxiliary_vs_normal_mean_difference = finiteMean(allD);
summary.auxiliary_winning_generation_rate = finiteMean(auxWinGeneration);
summary.auxiliary_pair_win_rate = safeRatio(sum(allPairWins),allPairTotal);
summary.auxiliary_parent_selection_rate = safeRatio(auxiliaryParents,auxiliaryCount);
summary.auxiliary_vs_direct_mean_difference = finiteMean(directDifferences);
summary.auxiliary_candidate_count = auxiliaryCount;
summary.branch_UFE = ins.UFEs; summary.branch_LFE = ins.LFEs;
summary.branch_UFE_delta = ins.UFEs-snapshot.UFE_before_sampling;
summary.branch_LFE_delta = ins.LFEs-snapshot.LFE_before_sampling;
summary.final_upper_error = abs(ins.UF-BI.u_fopt);
summary.final_lower_error = abs(ins.LF-BI.l_fopt);
summary.final_upper_success = double(summary.final_upper_error < BI.u_ftol);
summary.final_lower_success = double(summary.final_lower_error < BI.l_ftol);
summary.branch_termination = terminalReason(log, ...
    summary.completed_generations == summary.requested_horizon_generations);
end

function value = finiteMax(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = max(values); end
end

function value = finiteMean(values)
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = mean(values); end
end

function value = safeRatio(numerator,denominator)
if denominator <= 0, value = NaN; else, value = numerator/denominator; end
end

function weights = cmaWeights(populationSize)
mu = floor(populationSize/2);
weights = log(mu+1/2)-log(1:mu);
weights = weights/sum(weights);
end

function value = rowDistance(entry,candidateIndex,upperDim)
value = auxiliaryDistance(entry,candidateIndex,'actual',upperDim);
end

function value = auxiliaryDistance(entry,candidateIndex,kind,upperDim)
auxIndices = entry.auxiliary_indices(:)';
position = find(auxIndices == candidateIndex,1,'first');
if strcmp(kind,'raw')
    offsets = entry.auxiliary_upper_raw_offsets;
else
    offsets = entry.auxiliary_upper_actual_offsets;
end
if isempty(position) || size(offsets,1) < position
    value = NaN;
else
    value = norm(offsets(position,1:upperDim));
end
end

function text = auxiliaryJointText(entry,candidateIndex,kind)
auxIndices = entry.auxiliary_indices(:)';
position = find(auxIndices == candidateIndex,1,'first');
if isempty(position)
    text = '';
elseif strcmp(kind,'raw') && size(entry.auxiliary_raw_joint,1) >= position
    text = vectorText(entry.auxiliary_raw_joint(position,:));
elseif strcmp(kind,'actual') && size(entry.auxiliary_actual_joint,1) >= position
    text = vectorText(entry.auxiliary_actual_joint(position,:));
else
    text = '';
end
end

function value = relativeDistance(entry,candidateIndex)
auxIndices = entry.auxiliary_indices(:)';
position = find(auxIndices == candidateIndex,1,'first');
if isempty(position) || numel(entry.auxiliary_relative_distance_to_cma_scale) < position
    value = NaN;
else
    value = entry.auxiliary_relative_distance_to_cma_scale(position);
end
end

function value = candidateBoundaryRepair(entry,candidateIndex)
auxIndices = entry.auxiliary_indices(:)';
position = find(auxIndices == candidateIndex,1,'first');
if isempty(position) || size(entry.auxiliary_boundary_repaired,1) < position
    value = false;
else
    value = any(entry.auxiliary_boundary_repaired(position,:));
end
end

function reason = terminalReason(log,horizonCompleted)
if isempty(log)
    reason = 'no_branch_generation';
elseif horizonCompleted
    reason = 'horizon_completed';
else
    reason = log(end).termination_reason;
end
end

function aux = copyCommonAuxiliaryFields(aux,row)
names = intersect(fieldnames(aux),fieldnames(row));
for index = 1:numel(names), aux.(names{index}) = row.(names{index}); end
end

function label = etaLabel(eta)
if isnan(eta), label = 'A1'; else, label = sprintf('eta_%03d',round(100*eta)); end
end

function rows = selectContinuousScaleSnapshots(snapshots,terminalUFE)
rows = repmat(struct('selected',false,'state_index',NaN,'state_id','', ...
    'phase','','progress',NaN,'reason',''),numel(snapshots),1);
if isempty(snapshots), return; end
startUFE = snapshots(1).UFE_before_sampling;
denominator = terminalUFE-startUFE;
progress = NaN(numel(snapshots),1);
if denominator > 0
    progress = ([snapshots.UFE_before_sampling]'-startUFE)/denominator;
end
phases = {'early','middle','late'}; targets = [1/6,1/2,5/6];
for index = 1:numel(snapshots)
    rows(index).state_index = index;
    rows(index).state_id = sprintf('gen%04d',snapshots(index).generation);
    rows(index).progress = progress(index);
    rows(index).phase = phaseForProgress(progress(index));
    rows(index).reason = 'not_selected';
end
chosen = [];
for phaseIndex = 1:numel(phases)
    candidates = find(strcmp({rows.phase},phases{phaseIndex}));
    if isempty(candidates), continue; end
    % Phase midpoint is primary.  A small diversity term prefers a distinct
    % (d,q) state if several candidates are equally phase-representative.
    score = abs(progress(candidates)-targets(phaseIndex));
    if ~isempty(chosen)
        d0 = [snapshots(chosen).d]; q0 = [snapshots(chosen).q];
        for c = 1:numel(candidates)
            novelty = min(abs(snapshots(candidates(c)).d-d0) + ...
                abs(snapshots(candidates(c)).q-q0));
            score(c) = score(c) - 1e-6*novelty;
        end
    end
    [~,bestLocal] = min(score); best = candidates(bestLocal);
    rows(best).selected = true;
    rows(best).reason = 'phase_midpoint_with_dq_coverage';
    chosen(end+1) = best; %#ok<AGROW>
end
end

function phase = phaseForProgress(progress)
if ~isfinite(progress), phase = 'unclassified';
elseif progress < 1/3, phase = 'early';
elseif progress < 2/3, phase = 'middle';
else, phase = 'late';
end
end

function row = emptySnapshotRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_index',NaN, ...
    'state_id','','selected',NaN,'phase','','phase_progress',NaN, ...
    'selection_reason','','generation',NaN,'UFE_before_sampling',NaN, ...
    'LFE_before_sampling',NaN,'d',NaN,'q',NaN,'sigma',NaN,'sF',NaN, ...
    'model_scale',NaN,'model_design_condition',NaN,'model_hessian_rcond',NaN, ...
    'history_best_UF',NaN,'x_pred','','cma_mean','','Cuu','', ...
    'H_normalized','','H_physical','','raw_normal_joint','');
end

function row = emptyGenerationRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_index',NaN, ...
    'state_id','','phase','','eta',NaN,'eta_label','','repeat_no',NaN, ...
    'branch_seed',NaN,'generation',NaN,'generation_offset',NaN,'source','', ...
    'candidate_index',NaN,'UF',NaN,'UF_for_cma_update',NaN, ...
    'fitness_rank',NaN,'parent_selected',NaN, ...
    'parent_weight',NaN,'normal_best_UF',NaN,'direct_UF',NaN,'joint','', ...
    'joint_for_cma_update','', ...
    'prediction_point_upper','','prediction_point_UF',NaN, ...
    'cma_mean_before','','sigma_before',NaN,'Cuu_before','', ...
    'pc_before','','ps_before','', ...
    'cma_mean_after','','sigma_after',NaN,'C_after','', ...
    'prediction_valid',NaN,'injection_reason','','UFE',NaN,'LFE_total',NaN);
end

function row = emptyAuxiliaryRow()
row = emptyGenerationRow();
row.raw_joint = ''; row.actual_joint = '';
row.actual_joint_for_cma_update = '';
row.distance_to_prediction = NaN; row.relative_distance_to_cma_scale = NaN;
row.raw_distance_to_prediction = NaN;
row.auxiliary_sampling_eta = NaN; row.auxiliary_sigma = NaN;
row.auxiliary_rms_search_scale = NaN;
row.boundary_repaired = NaN; row.better_than_direct = NaN;
row.better_than_normal_best = NaN; row.pair_wins = NaN; row.pair_total = NaN;
row.history_best_before_candidate = NaN; row.history_refresh = NaN;
row.history_improvement = NaN;
end

function row = emptyBranchSummaryRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_index',NaN, ...
    'state_id','','phase','','eta',NaN,'eta_label','','repeat_no',NaN, ...
    'branch_seed',NaN,'start_generation',NaN, ...
    'requested_horizon_generations',NaN,'completed_generations',NaN, ...
    'early_termination',NaN, ...
    'valid_prediction_generations',NaN,'prediction_fallback_generations',NaN, ...
    'd',NaN,'q',NaN,'sigma_start',NaN,'sF',NaN, ...
    'history_best_start_UF',NaN,'history_best_end_UF',NaN, ...
    'overall_history_improvement',NaN,'auxiliary_best_UF',NaN, ...
    'auxiliary_improvement_over_start',NaN, ...
    'auxiliary_history_refresh_count',NaN, ...
    'auxiliary_history_refresh_rate',NaN, ...
    'auxiliary_refresh_improvement_total',NaN, ...
    'auxiliary_vs_normal_mean_difference',NaN, ...
    'auxiliary_winning_generation_rate',NaN, ...
    'auxiliary_pair_win_rate',NaN, ...
    'auxiliary_parent_selection_rate',NaN, ...
    'auxiliary_vs_direct_mean_difference',NaN, ...
    'auxiliary_candidate_count',NaN,'branch_UFE',NaN,'branch_LFE',NaN, ...
    'branch_UFE_delta',NaN,'branch_LFE_delta',NaN, ...
    'final_upper_error',NaN,'final_lower_error',NaN, ...
    'final_upper_success',NaN,'final_lower_success',NaN, ...
    'branch_termination','','branch_runtime_seconds',NaN, ...
    'main_UFE_end',NaN,'main_LFE_end',NaN);
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
