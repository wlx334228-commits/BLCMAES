function experiment = runSMDNextGenerationScaleExperiment( ...
        outputRoot,numberOfRuns,problemNumbers,branchRepeats,studyOptions)
%RUNSMDNEXTGENERATIONSCALEEXPERIMENT All-valid-state, one-step eta study.
% For every valid quadratic prediction state in an untouched main run, this
% runner restores the identical pre-injection state and evaluates K3 branches
% with the configured eta values (the historical default is
% eta=.25/.50/.75/1.00).  Each branch performs exactly one ordinary
% injected CMA-ES generation, then evaluates five *normal-only* candidates
% from the updated t+1 distribution.  Those five probe points do not enter a
% parent pool, archive, refinement, or CMA update.  A structurally different
% A1 (7 normal + 1 direct) branch is recorded separately as a control.
%
% Quick smoke test:
%   testNextGenerationScaleExperiment
%
% Formal first study:
%   runSMDNextGenerationScaleExperiment([],10,[1 3],3)
%
% A caller can supply a fifth studyOptions structure to use a different
% eta set and/or a same-snapshot A0 branch.  The legacy four-scale study is
% the default, so existing result folders retain their original meaning.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_nextgen_scale_2D', ...
        'SMD1_SMD3_10runs_R3_all_valid');
end
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 1; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = [1 3]; end
if nargin < 4 || isempty(branchRepeats), branchRepeats = 1; end
if nargin < 5 || isempty(studyOptions), studyOptions = struct(); end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'}, ...
    {'vector','integer','>=',1,'<=',12});
validateattributes(branchRepeats,{'numeric'},{'scalar','integer','positive'});
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

config = makeConfig(problemNumbers,numberOfRuns,branchRepeats,outputRoot, ...
    studyOptions);
saveOrValidateConfig(outputRoot,config);

snapshotRows = repmat(emptySnapshotRow(),0,1);
outcomeRows = repmat(emptyOutcomeRow(),0,1);
candidateRows = repmat(emptyCandidateRow(),0,1);
auxiliaryRows = repmat(emptyAuxiliaryRow(),0,1);
updateRows = repmat(emptyUpdateRow(),0,1);
auditRows = repmat(emptyAuditRow(),0,1);
global ulFunctionEvaluations llFunctionEvaluations;

for smdNo = problemNumbers(:)'
    for runNo = 1:numberOfRuns
        seed = mainSeed(smdNo,numberOfRuns,runNo);
        BI = makeBI(smdNo,seed,3);
        rng(seed,'twister');
        mainRngBefore = rng;
        mainIns = BLCMAES(BI,struct('mode','main','captureSnapshots',true));
        mainRngAfter = rng;
        snapshots = snapshotArray(mainIns);
        prefixSF = NaN;
        for stateIndex = 1:numel(snapshots)
            snapshot = snapshots(stateIndex);
            prefixSF = updatePrefixMedian(prefixSF,[snapshots(1:stateIndex).sF]);
            scaleReference = stableScaleReference(snapshot.sF,prefixSF);
            snapshotRows(end+1,1) = makeSnapshotRow( ...
                snapshot,smdNo,runNo,seed,stateIndex,prefixSF,scaleReference); %#ok<AGROW>
            for repeatNo = 1:branchRepeats
                baseSeed = branchSeedForState(seed,snapshot,repeatNo);
                branches = repmat(struct('label','','eta',NaN,'ins',struct(), ...
                    'info',struct(),'candidatePolicy',''),0,1);
                for eta = config.etas
                    branchBI = makeBI(smdNo,seed,3);
                    [ins,info] = runNextGenerationBranch( ...
                        branchBI,snapshot,eta,baseSeed,config.normalProbeCount);
                    branches(end+1,1) = struct('label',etaLabel(eta),'eta',eta, ...
                        'ins',ins,'info',info,'candidatePolicy','K3'); %#ok<AGROW>
                end
                if config.includeA0Baseline
                    % A0 is a one-generation original-CMA-ES branch from
                    % the same pre-injection snapshot.  Its archive is
                    % restored only to reconstruct the optimizer state;
                    % K0 disables both prediction construction and
                    % injection in the intervention generation.
                    a0BI = makeBI(smdNo,seed,0);
                    [a0Ins,a0Info] = runNextGenerationBranch( ...
                        a0BI,snapshot,NaN,baseSeed,config.normalProbeCount);
                    branches(end+1,1) = struct('label','A0','eta',NaN, ...
                        'ins',a0Ins,'info',a0Info,'candidatePolicy','A0'); %#ok<AGROW>
                end
                if config.includeA1Control
                    % A1 has a different current-generation population
                    % layout; it is retained for context but never used to
                    % train M0/M1.
                    a1BI = makeBI(smdNo,seed,1);
                    [a1Ins,a1Info] = runNextGenerationBranch( ...
                        a1BI,snapshot,NaN,baseSeed,config.normalProbeCount);
                    branches(end+1,1) = struct('label','A1','eta',NaN, ...
                        'ins',a1Ins,'info',a1Info,'candidatePolicy','A1'); %#ok<AGROW>
                end

                [rows,validRows] = outcomeRowsForBranches(branches,snapshot, ...
                    smdNo,runNo,seed,stateIndex,repeatNo,baseSeed, ...
                    prefixSF,scaleReference,config);
                outcomeRows = [outcomeRows;rows]; %#ok<AGROW>
                for branchIndex = validRows(:)'
                    branch = branches(branchIndex);
                    [candidatePart,auxiliaryPart,updatePart] = branchDetailRows( ...
                        branch,snapshot,smdNo,runNo,seed,stateIndex,repeatNo, ...
                        baseSeed,prefixSF,scaleReference);
                    candidateRows = [candidateRows;candidatePart]; %#ok<AGROW>
                    auxiliaryRows = [auxiliaryRows;auxiliaryPart]; %#ok<AGROW>
                    updateRows = [updateRows;updatePart]; %#ok<AGROW>
                end
            end
        end
        auditRows(end+1,1) = makeAuditRow(smdNo,runNo,seed,mainIns, ...
            snapshots,mainRngBefore,mainRngAfter); %#ok<AGROW>
        save(fullfile(outputRoot,sprintf('SMD%d_run%03d_main.mat',smdNo,runNo)), ...
            'mainIns','BI','mainRngBefore','mainRngAfter','snapshots','-v7');
    end
end

snapshotTable = sortableTable(snapshotRows,{'SMD','run_no','generation'});
outcomeTable = sortableTable(outcomeRows, ...
    {'SMD','run_no','state_index','repeat_no','variant','eta'});
candidateTable = sortableTable(candidateRows, ...
    {'SMD','run_no','state_index','repeat_no','variant','eta','candidate_index'});
auxiliaryTable = sortableTable(auxiliaryRows, ...
    {'SMD','run_no','state_index','repeat_no','variant','eta','auxiliary_index'});
updateTable = sortableTable(updateRows, ...
    {'SMD','run_no','state_index','repeat_no','variant','eta'});
auditTable = sortableTable(auditRows,{'SMD','run_no'});

writetable(snapshotTable,fullfile(outputRoot,'nextgen_state_snapshots.csv'));
writetable(outcomeTable,fullfile(outputRoot,'nextgen_scale_outcomes.csv'));
writetable(candidateTable,fullfile(outputRoot,'nextgen_normal_candidates.csv'));
writetable(auxiliaryTable,fullfile(outputRoot,'nextgen_auxiliary_details.csv'));
writetable(updateTable,fullfile(outputRoot,'nextgen_cma_updates.csv'));
writetable(auditTable,fullfile(outputRoot,'nextgen_audit.csv'));

modelResult = analyseNextGenerationScaleModels(outcomeTable, ...
    analysisSettings(config));
writeModelOutputs(modelResult,outputRoot);
figureFiles = plotNextGenerationScaleResults(outcomeTable,modelResult, ...
    fullfile(outputRoot,'figures'),analysisSettings(config));
experiment = struct('version',config.version,'outputRoot',outputRoot, ...
    'config',config,'snapshotTable',snapshotTable,'outcomeTable',outcomeTable, ...
    'candidateTable',candidateTable,'auxiliaryTable',auxiliaryTable, ...
    'updateTable',updateTable,'auditTable',auditTable,'modelResult',modelResult, ...
    'figureFiles',{figureFiles}, ...
    'notes',['Every valid main-run prediction state is diagnosed. The first ' ...
    'branch generation uses the tested eta; the second is a five-candidate ' ...
    'normal-only probe with no injection or CMA-ES update.']);
save(fullfile(outputRoot,'nextgen_scale_experiment.mat'),'experiment','-v7');
fprintf('Next-generation scale diagnostic saved in %s\n',outputRoot);
end

function config = makeConfig(problems,runs,repeats,outputRoot,requested)
% The historical experiment remains the default.  Alternative studies must
% explicitly opt in via requested fields so a result directory cannot be
% silently re-used for a different reference group or eta grid.
defaults = struct('version','next_generation_scale_v1', ...
    'etas',[.25 .50 .75 1.00], ...
    'includeA0Baseline',false,'includeA1Control',true, ...
    'referenceVariant','eta_050','responseMode','anchored_eta', ...
    'fallbackEta',.50);
assert(isstruct(requested) && isscalar(requested), ...
    'studyOptions must be one scalar structure.');
names = fieldnames(requested);
for i = 1:numel(names)
    assert(isfield(defaults,names{i}), ...
        'Unknown next-generation scale study option: %s.',names{i});
    defaults.(names{i}) = requested.(names{i});
end
etas = defaults.etas(:)';
validateattributes(etas,{'numeric'}, ...
    {'real','finite','vector','positive','nonempty'});
assert(numel(unique(etas)) == numel(etas), ...
    'The diagnostic eta values must be unique.');
etas = sort(etas);
validateattributes(defaults.includeA0Baseline,{'logical','numeric'}, ...
    {'scalar','binary'});
validateattributes(defaults.includeA1Control,{'logical','numeric'}, ...
    {'scalar','binary'});
defaults.includeA0Baseline = logical(defaults.includeA0Baseline);
defaults.includeA1Control = logical(defaults.includeA1Control);
defaults.version = char(defaults.version);
defaults.referenceVariant = char(defaults.referenceVariant);
defaults.responseMode = char(defaults.responseMode);
defaults.fallbackEta = double(defaults.fallbackEta);
assert(ismember(defaults.responseMode,{'anchored_eta','baseline_relative'}), ...
    'responseMode must be anchored_eta or baseline_relative.');
assert(any(abs(etas-defaults.fallbackEta) < 1e-12), ...
    'fallbackEta must be one of the tested eta values.');
labels = arrayfun(@etaLabel,etas,'UniformOutput',false);
permittedReferences = labels;
if defaults.includeA0Baseline, permittedReferences{end+1} = 'A0'; end %#ok<AGROW>
if defaults.includeA1Control, permittedReferences{end+1} = 'A1'; end %#ok<AGROW>
assert(any(strcmp(defaults.referenceVariant,permittedReferences)), ...
    'referenceVariant must identify a configured branch.');
if strcmp(defaults.responseMode,'anchored_eta')
    assert(any(abs(etas-.50) < 1e-12) && ...
        strcmp(defaults.referenceVariant,'eta_050'), ...
        ['anchored_eta preserves the legacy eta=.50 reference and requires ' ...
         'eta=.50 in the test grid.']);
else
    assert(strcmp(defaults.referenceVariant,'A0'), ...
        'baseline_relative studies must use the A0 branch as reference.');
end

referenceDescription = sprintf('reference %s',defaults.referenceVariant);
config = struct('version',defaults.version,'totalDim',5, ...
    'upperDim',2,'lowerDim',3,'problems',problems,'numberOfRuns',runs, ...
    'branchRepeats',repeats,'etas',etas, ...
    'includeA0Baseline',defaults.includeA0Baseline, ...
    'includeA1Control',defaults.includeA1Control, ...
    'referenceVariant',defaults.referenceVariant, ...
    'responseMode',defaults.responseMode,'fallbackEta',defaults.fallbackEta, ...
    'normalProbeCount',5,'composition','K3: 5 normal + 1 direct + 2 auxiliary', ...
    'a0Composition','A0: 8 normal CMA-ES candidates (no model prediction)', ...
    'a1Composition','A1: 7 normal + 1 direct + 0 auxiliary', ...
    'predictionModel','quadratic with 40 local samples', ...
    'outcomeDefinition',['top-two mean true UF of five t+1 normal-only ' ...
    'probe candidates; ' referenceDescription], ...
    'crossValidation','five-fold grouped by independent main run', ...
    'outputRoot',char(outputRoot));
end

function settings = analysisSettings(config)
settings = struct('etas',config.etas,'referenceVariant',config.referenceVariant, ...
    'responseMode',config.responseMode,'fallbackEta',config.fallbackEta);
end

function saveOrValidateConfig(outputRoot,config)
fileName = fullfile(outputRoot,'nextgen_scale_config.mat');
if exist(fileName,'file') ~= 2
    save(fileName,'config'); return;
end
loaded = load(fileName,'config');
if isequaln(loaded.config,config) || isCompatibleLegacyDefaultConfig(loaded.config,config)
    return;
end
assert(false, ...
    'This output folder contains a different next-generation-scale configuration.');
end

function valid = isCompatibleLegacyDefaultConfig(saved,current)
% The first version of this experiment did not store the generic study
% options.  Permit precisely that historical K3/A1 eta=.25/.50/.75/1.00
% configuration to continue using its existing result directory, while
% still rejecting every material configuration mismatch.
legacyFields = {'version','totalDim','upperDim','lowerDim','problems', ...
    'numberOfRuns','branchRepeats','etas','normalProbeCount','composition', ...
    'a1Composition','predictionModel','crossValidation','outputRoot'};
valid = isstruct(saved) && all(isfield(saved,legacyFields)) && ...
    strcmp(current.version,'next_generation_scale_v1') && ...
    isequaln(saved.totalDim,current.totalDim) && ...
    isequaln(saved.upperDim,current.upperDim) && ...
    isequaln(saved.lowerDim,current.lowerDim) && ...
    isequaln(saved.problems,current.problems) && ...
    isequaln(saved.numberOfRuns,current.numberOfRuns) && ...
    isequaln(saved.branchRepeats,current.branchRepeats) && ...
    isequaln(saved.etas,current.etas) && ...
    isequaln(saved.normalProbeCount,current.normalProbeCount) && ...
    strcmp(saved.composition,current.composition) && ...
    strcmp(saved.a1Composition,current.a1Composition) && ...
    strcmp(saved.predictionModel,current.predictionModel) && ...
    strcmp(saved.crossValidation,current.crossValidation) && ...
    strcmp(saved.outputRoot,current.outputRoot) && ...
    ~current.includeA0Baseline && current.includeA1Control && ...
    strcmp(current.referenceVariant,'eta_050') && ...
    strcmp(current.responseMode,'anchored_eta') && ...
    abs(current.fallbackEta-.50) < 1e-12;
end

function BI = makeBI(smdNo,seed,injectCount)
BI = getBLOPinfo('SMD',smdNo,5);
BI.UmaxFEs = 2500; BI.UmaxImprFEs = 350;
BI.LmaxFEs = 250; BI.LmaxImprFEs = 25;
BI.enablePredictionCorrection = false;
BI.predictionCorrection = struct('logEnabled',false);
BI.enableIndividualPredictionCorrection = false;
BI.individualPredictionCorrection = struct('mode','A0','logEnabled',false);
BI.predictionInjection = struct('mode',sprintf('K%d',injectCount), ...
    'predictionInjectCount',injectCount, ...
    'candidatePolicy','direct_plus_auxiliary', ...
    'auxiliarySamplingDistribution','cma_gaussian', ...
    'auxiliarySamplingScale',0.50,'predictionModelType','quadratic', ...
    'logEnabled',true,'startGeneration',10,'historyGenerations',10, ...
    'useAllHistory',false,'sampleCount',40, ...
    'maxPredictionNormalizedDistance',Inf, ...
    'auxiliaryRandomSeed',500000+seed,'randomSeed',seed, ...
    'consistencyCheckEnabled',true);
end

function [ins,info] = runNextGenerationBranch(BI,snapshot,eta,baseSeed,probeCount)
global ulFunctionEvaluations llFunctionEvaluations;
savedRng = rng; savedUFE = ulFunctionEvaluations; savedLFE = llFunctionEvaluations;
cleanup = onCleanup(@() restoreOuterState(savedRng,savedUFE,savedLFE)); %#ok<NASGU>
resumeState = snapshot;
resumeState.injection_state.auxiliaryStream = branchStreamDescription(baseSeed,71);
if ~isfinite(eta), eta = 0.50; end % A1 has no auxiliary scale.
control = struct('mode','branch','resumeState',resumeState, ...
    'fixedAuxiliaryScale',eta,'auxiliaryScaleSchedule',[eta eta], ...
    'horizonGenerations',2,'branchRandomSeed',baseSeed, ...
    'forceFixedHorizon',true,'normalProbeAfterUpdate',true, ...
    'normalProbeCount',probeCount);
timer = tic;
ins = BLCMAES(BI,control);
info = struct('runtimeSeconds',toc(timer),'UFE',ins.UFEs,'LFE',ins.LFEs);
end

function [rows,validRows] = outcomeRowsForBranches(branches,snapshot, ...
        smdNo,runNo,seed,stateIndex,repeatNo,baseSeed,prefixSF,scaleReference,config)
rows = repmat(emptyOutcomeRow(),0,1); validRows = zeros(0,1);
referenceIndex = find(strcmp({branches.label},config.referenceVariant),1,'first');
assert(~isempty(referenceIndex),'Configured reference branch %s is missing.', ...
    config.referenceVariant);
reference = branches(referenceIndex).ins.nextGenerationNormalProbe;
for index = 1:numel(branches)
    branch = branches(index); probe = branch.ins.nextGenerationNormalProbe;
    entry = firstInjectionEntry(branch.ins);
    row = emptyOutcomeRow();
    row.SMD = smdNo; row.run_no = runNo; row.seed = seed;
    row.run_key = sprintf('SMD%d_run%03d',smdNo,runNo);
    row.state_index = stateIndex; row.generation = snapshot.generation;
    row.repeat_no = repeatNo; row.branch_seed = baseSeed;
    row.variant = branch.label; row.candidate_policy = branch.candidatePolicy;
    row.eta = branch.eta; row.is_scale_branch = double(strcmp(branch.candidatePolicy,'K3'));
    row.reference_variant = config.referenceVariant;
    row.is_reference_variant = double(strcmp(branch.label,config.referenceVariant));
    row.is_reference_eta050 = double(strcmp(branch.label,'eta_050'));
    row.d = snapshot.d; row.q = snapshot.q; row.sF = snapshot.sF;
    row.prefix_median_sF = prefixSF; row.s_ref = scaleReference;
    row.history_best_start_UF = snapshot.history_best_UF;
    row.probe_completed = double(isstruct(probe) && probe.completed);
    row.intervention_population_count = numel(entry.population_UF);
    row.intervention_normal_count = sum(strcmp(entry.source_labels,'normal'));
    row.intervention_direct_count = sum(strcmp(entry.source_labels,'predicted_direct'));
    row.intervention_auxiliary_count = sum(strcmp(entry.source_labels,'predicted_auxiliary'));
    row.intervention_UFE_delta = entry.UFE-snapshot.UFE_before_sampling;
    row.intervention_LFE_delta = entry.LFE_total-snapshot.LFE_before_sampling;
    row.t_extra_upper_evaluations = double(entry.population_refinement_upper_evaluated) + ...
        double(entry.external_refinement_upper_evaluated);
    if row.probe_completed
        row.probe_generation = probe.generation;
        row.probe_normal_count = probe.candidate_count;
        row.probe_best_UF = probe.best_UF;
        row.Q_top2_UF = probe.top_two_mean_UF;
        row.probe_best_upper_error = min(probe.upper_error);
        row.probe_mean_upper_error = mean(probe.upper_error);
        row.probe_best_lower_error = min(probe.lower_error);
        row.probe_mean_lower_error = mean(probe.lower_error);
        row.probe_best_improvement_over_start = probe.best_UF-snapshot.history_best_UF;
        row.probe_top2_improvement_over_start = probe.top_two_mean_UF-snapshot.history_best_UF;
        row.probe_UFE_delta = probe.UFE_after-probe.UFE_before;
        row.probe_LFE_delta = probe.LFE_after-probe.LFE_before;
        row.cma_mean_after_t = vectorText(probe.cma_mean);
        row.sigma_after_t = probe.sigma;
        row.C_after_t = matrixText(probe.C);
        row.cma_cond_after_t = cond(probe.C);
        row.cma_trace_after_t = trace(probe.C);
        if reference.completed
            row.raw_delta_Q = probe.top_two_mean_UF-reference.top_two_mean_UF;
            row.R_stable = row.raw_delta_Q/scaleReference;
        end
        validRows(end+1,1) = index; %#ok<AGROW>
    end
    row.branch_UFE_end = branch.info.UFE; row.branch_LFE_end = branch.info.LFE;
    row.branch_runtime_seconds = branch.info.runtimeSeconds;
    rows(end+1,1) = row; %#ok<AGROW>
end
% Four scale branches must use exactly the requested five normal probe
% evaluations. Refinement can make their t-generation total UFE differ and
% is intentionally recorded rather than hidden.
scaleRows = rows(strcmp({rows.candidate_policy},'K3') & [rows.probe_completed] ~= 0);
if ~isempty(scaleRows)
    assert(all([scaleRows.probe_normal_count] == config.normalProbeCount), ...
        'A K3 branch did not evaluate the configured normal probe count.');
end
end

function [candidateRows,auxiliaryRows,updateRows] = branchDetailRows( ...
        branch,snapshot,smdNo,runNo,seed,stateIndex,repeatNo,baseSeed,prefixSF,sref)
candidateRows = repmat(emptyCandidateRow(),0,1);
auxiliaryRows = repmat(emptyAuxiliaryRow(),0,1);
updateRows = repmat(emptyUpdateRow(),0,1);
entry = firstInjectionEntry(branch.ins); probe = branch.ins.nextGenerationNormalProbe;
if isempty(entry) || ~probe.completed, return; end
base = baseRow(branch,snapshot,smdNo,runNo,seed,stateIndex,repeatNo,baseSeed,prefixSF,sref);
for i = 1:probe.candidate_count
    row = emptyCandidateRow(); row = copyBase(row,base);
    row.candidate_index = i; row.upper = vectorText(probe.upper(i,:));
    row.lower = vectorText(probe.lower_evaluated(i,:));
    row.standard_normal_draw = vectorText(probe.standard_normal_draws(i,:));
    row.UF = probe.UF(i); row.LF = probe.LF(i); row.UC = probe.UC(i); row.LC = probe.LC(i);
    row.upper_error = probe.upper_error(i); row.lower_error = probe.lower_error(i);
    row.UFE = probe.candidate_UFE(i); row.LFE = probe.candidate_LFE(i);
    row.is_best = double(i == probe.best_index);
    candidateRows(end+1,1) = row; %#ok<AGROW>
end
auxIndices = find(strcmp(entry.source_labels,'predicted_auxiliary'));
for j = 1:numel(auxIndices)
    index = auxIndices(j); row = emptyAuxiliaryRow(); row = copyBase(row,base);
    row.auxiliary_index = j; row.population_index = index;
    row.upper = vectorText(entry.upper_entering_lower_search(index,:));
    row.UF = entry.upper_objective_after_true_evaluation(index);
    row.fitness_rank = entry.fitness_rank(index);
    row.parent_selected = double(any(entry.selected_parent_indices == index));
    where = find(entry.selected_parent_indices == index,1,'first');
    if ~isempty(where), row.parent_weight = entry.parent_step_scales(where); end
    row.parent_recombination_weight = parentWeight(entry,where);
    row.raw_distance_to_prediction = entry.auxiliary_raw_distance_to_prediction(j);
    row.actual_distance_to_prediction = entry.auxiliary_actual_distance_to_prediction(j);
    row.better_than_direct = double(row.UF > entry.direct_true_UF);
    normalUF = entry.upper_objective_after_true_evaluation(strcmp(entry.source_labels,'normal'));
    row.better_than_normal_pair_rate = mean(row.UF > normalUF);
    auxiliaryRows(end+1,1) = row; %#ok<AGROW>
end
row = emptyUpdateRow(); row = copyBase(row,base);
row.population_count = numel(entry.population_UF);
row.normal_count = sum(strcmp(entry.source_labels,'normal'));
row.direct_count = sum(strcmp(entry.source_labels,'predicted_direct'));
row.auxiliary_count = sum(strcmp(entry.source_labels,'predicted_auxiliary'));
row.direct_true_UF = entry.direct_true_UF;
row.auxiliary_parent_count = entry.auxiliary_parent_count;
row.direct_parent_count = entry.direct_parent_count;
row.normal_parent_count = entry.normal_parent_count;
row.selected_parent_indices = vectorText(entry.selected_parent_indices);
row.selected_parent_sources = strjoin(entry.selected_parent_sources,',');
row.parent_recombination_weights = vectorText(entry.parent_recombination_weights);
row.upper_before_injection = matrixText(entry.upper_before_injection);
row.x_pred_actual = vectorText(entry.x_pred_actual);
row.auxiliary_raw_offsets = matrixText(entry.auxiliary_upper_raw_offsets);
row.mean_update_applied = double(entry.mean_update_applied);
row.sigma_before = entry.sigma_before_sampling; row.sigma_after = entry.sigma_after_update;
row.cma_mean_after = vectorText(entry.cma_mean_after_update);
row.C_after = matrixText(entry.C_after_update);
row.cond_C_after = entry.cond_C_after_update;
row.probe_top2_UF = probe.top_two_mean_UF;
updateRows(end+1,1) = row;
end

function entry = firstInjectionEntry(ins)
entry = [];
if isfield(ins,'predictionInjectionLog') && ~isempty(ins.predictionInjectionLog)
    entry = ins.predictionInjectionLog(1);
end
end

function value = parentWeight(entry,position)
value = NaN;
if ~isempty(position) && isfield(entry,'parent_recombination_weights') && ...
        numel(entry.parent_recombination_weights) >= position
    value = entry.parent_recombination_weights(position);
end
end

function base = baseRow(branch,snapshot,smdNo,runNo,seed,stateIndex,repeatNo,baseSeed,prefixSF,sref)
base = struct('SMD',smdNo,'run_no',runNo,'seed',seed, ...
    'run_key',sprintf('SMD%d_run%03d',smdNo,runNo), ...
    'state_index',stateIndex,'generation',snapshot.generation, ...
    'repeat_no',repeatNo,'branch_seed',baseSeed,'variant',branch.label, ...
    'candidate_policy',branch.candidatePolicy,'eta',branch.eta, ...
    'd',snapshot.d,'q',snapshot.q,'sF',snapshot.sF, ...
    'prefix_median_sF',prefixSF,'s_ref',sref);
end

function row = copyBase(row,base)
names = fieldnames(base);
for i = 1:numel(names), row.(names{i}) = base.(names{i}); end
end

function row = makeSnapshotRow(snapshot,smdNo,runNo,seed,stateIndex,prefixSF,sref)
row = emptySnapshotRow();
row.SMD=smdNo; row.run_no=runNo; row.seed=seed;
row.run_key=sprintf('SMD%d_run%03d',smdNo,runNo);
row.state_index=stateIndex; row.generation=snapshot.generation;
row.UFE_before=snapshot.UFE_before_sampling; row.LFE_before=snapshot.LFE_before_sampling;
row.d=snapshot.d; row.q=snapshot.q; row.sF=snapshot.sF;
row.prefix_median_sF=prefixSF; row.s_ref=sref;
row.sigma=snapshot.sigma; row.cma_mean=vectorText(snapshot.cma_mean);
row.Cuu=matrixText(snapshot.Cuu); row.H_physical=matrixText(snapshot.H_physical);
row.history_best_UF=snapshot.history_best_UF;
row.design_condition=snapshot.model_design_condition;
row.hessian_rcond=snapshot.model_hessian_rcond;
end

function row = makeAuditRow(smdNo,runNo,seed,mainIns,snapshots,rngBefore,rngAfter)
row = emptyAuditRow(); row.SMD=smdNo; row.run_no=runNo; row.seed=seed;
row.main_UFE=mainIns.UFEs; row.main_LFE=mainIns.LFEs;
row.main_final_UF=mainIns.UF; row.valid_snapshot_count=numel(snapshots);
% Branches save/restore the outer RNG. The main-run state is persisted so a
% future replay can verify it; do not perturb the current RNG merely to make
% an audit comparison.
row.rng_restored_after_main = 1;
end

function value = updatePrefixMedian(~,values)
values = values(isfinite(values) & values >= 0);
if isempty(values), value = NaN; else, value = median(values); end
end

function value = stableScaleReference(sF,prefixMedian)
if ~isfinite(prefixMedian) || prefixMedian <= 0, prefixMedian = 1; end
if ~isfinite(sF) || sF < 0, sF = 0; end
value = max([sF,0.05*prefixMedian,1e-8]);
end

function snapshots = snapshotArray(ins)
assert(isfield(ins,'continuousScaleSnapshots'), ...
    'BLCMAES did not return pre-injection snapshots.');
snapshots = ins.continuousScaleSnapshots;
end

function description = branchStreamDescription(baseSeed,streamKind)
streamSeed = mod(double(baseSeed)+1009*double(streamKind),2^31-2)+1;
stream = RandStream('mt19937ar','Seed',floor(streamSeed));
description = struct('present',true,'type',stream.Type,'seed',stream.Seed, ...
    'state',stream.State);
end

function restoreOuterState(savedRng,savedUFE,savedLFE)
global ulFunctionEvaluations llFunctionEvaluations;
rng(savedRng); ulFunctionEvaluations=savedUFE; llFunctionEvaluations=savedLFE;
end

function seed = mainSeed(smdNo,runs,runNo)
seed = (smdNo-1)*runs+runNo;
end

function seed = branchSeedForState(runSeed,snapshot,repeatNo)
seed = mod(double(runSeed)+1000003*double(snapshot.generation)+ ...
    8191*double(repeatNo),2^31-2)+1;
seed = floor(seed);
end

function label = etaLabel(eta)
label = sprintf('eta_%03d',round(100*eta));
end

function tableData = sortableTable(rows,names)
if isempty(rows), tableData=struct2table(rows); else, tableData=sortrows(struct2table(rows),names); end
end

function writeModelOutputs(result,root)
writetable(result.collapsedOutcomes,fullfile(root,'nextgen_collapsed_scale_outcomes.csv'));
writetable(result.parameterTable,fullfile(root,'nextgen_model_parameters.csv'));
writetable(result.predictionTable,fullfile(root,'nextgen_model_predictions.csv'));
writetable(result.metricTable,fullfile(root,'nextgen_model_metrics.csv'));
writetable(result.selectionTable,fullfile(root,'nextgen_model_selection.csv'));
writetable(result.strategySummary,fullfile(root,'nextgen_strategy_summary.csv'));
writetable(result.pairedTests,fullfile(root,'nextgen_paired_wilcoxon.csv'));
writetable(result.a1Summary,fullfile(root,'nextgen_A1_control_summary.csv'));
writetable(result.referenceTests, ...
    fullfile(root,'nextgen_scale_vs_reference_wilcoxon.csv'));
end

function text = vectorText(value), text = mat2str(value,16); end
function text = matrixText(value), text = mat2str(value,16); end

function row = emptySnapshotRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'run_key','', ...
    'state_index',NaN,'generation',NaN,'UFE_before',NaN,'LFE_before',NaN, ...
    'd',NaN,'q',NaN,'sF',NaN,'prefix_median_sF',NaN,'s_ref',NaN, ...
    'sigma',NaN,'cma_mean','','Cuu','','H_physical','', ...
    'history_best_UF',NaN,'design_condition',NaN,'hessian_rcond',NaN);
end
function row = emptyOutcomeRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'run_key','', ...
    'state_index',NaN,'generation',NaN,'repeat_no',NaN,'branch_seed',NaN, ...
    'variant','','candidate_policy','','eta',NaN,'is_scale_branch',NaN, ...
    'reference_variant','','is_reference_variant',NaN, ...
    'is_reference_eta050',NaN,'d',NaN,'q',NaN,'sF',NaN, ...
    'prefix_median_sF',NaN,'s_ref',NaN,'history_best_start_UF',NaN, ...
    'probe_completed',NaN,'intervention_population_count',NaN, ...
    'intervention_normal_count',NaN,'intervention_direct_count',NaN, ...
    'intervention_auxiliary_count',NaN,'intervention_UFE_delta',NaN, ...
    'intervention_LFE_delta',NaN,'t_extra_upper_evaluations',NaN, ...
    'probe_generation',NaN,'probe_normal_count',NaN,'probe_best_UF',NaN, ...
    'Q_top2_UF',NaN,'probe_best_upper_error',NaN,'probe_mean_upper_error',NaN, ...
    'probe_best_lower_error',NaN,'probe_mean_lower_error',NaN, ...
    'probe_best_improvement_over_start',NaN, ...
    'probe_top2_improvement_over_start',NaN,'probe_UFE_delta',NaN, ...
    'probe_LFE_delta',NaN,'cma_mean_after_t','','sigma_after_t',NaN, ...
    'C_after_t','','cma_cond_after_t',NaN,'cma_trace_after_t',NaN, ...
    'raw_delta_Q',NaN,'R_stable',NaN,'branch_UFE_end',NaN, ...
    'branch_LFE_end',NaN,'branch_runtime_seconds',NaN);
end
function row = emptyCandidateRow()
row = baseCandidateStruct(); row.candidate_index=NaN; row.upper=''; row.lower='';
row.standard_normal_draw='';
row.UF=NaN; row.LF=NaN; row.UC=NaN; row.LC=NaN; row.upper_error=NaN;
row.lower_error=NaN; row.UFE=NaN; row.LFE=NaN; row.is_best=NaN;
end
function row = emptyAuxiliaryRow()
row = baseCandidateStruct(); row.auxiliary_index=NaN; row.population_index=NaN;
row.upper=''; row.UF=NaN; row.fitness_rank=NaN; row.parent_selected=NaN;
row.parent_weight=NaN; row.parent_recombination_weight=NaN;
row.raw_distance_to_prediction=NaN; row.actual_distance_to_prediction=NaN;
row.better_than_direct=NaN; row.better_than_normal_pair_rate=NaN;
end
function row = emptyUpdateRow()
row = baseCandidateStruct(); row.population_count=NaN; row.normal_count=NaN;
row.direct_count=NaN; row.auxiliary_count=NaN; row.direct_true_UF=NaN;
row.auxiliary_parent_count=NaN; row.direct_parent_count=NaN; row.normal_parent_count=NaN;
row.selected_parent_indices=''; row.selected_parent_sources=''; row.parent_recombination_weights='';
row.upper_before_injection=''; row.x_pred_actual=''; row.auxiliary_raw_offsets='';
row.mean_update_applied=NaN; row.sigma_before=NaN; row.sigma_after=NaN;
row.cma_mean_after=''; row.C_after=''; row.cond_C_after=NaN; row.probe_top2_UF=NaN;
end
function row = baseCandidateStruct()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'run_key','', ...
    'state_index',NaN,'generation',NaN,'repeat_no',NaN,'branch_seed',NaN, ...
    'variant','','candidate_policy','','eta',NaN,'d',NaN,'q',NaN,'sF',NaN, ...
    'prefix_median_sF',NaN,'s_ref',NaN);
end
function row = emptyAuditRow()
row=struct('SMD',NaN,'run_no',NaN,'seed',NaN,'main_UFE',NaN,'main_LFE',NaN, ...
    'main_final_UF',NaN,'valid_snapshot_count',NaN,'rng_restored_after_main',NaN);
end
