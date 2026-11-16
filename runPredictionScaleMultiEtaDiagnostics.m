function experiment = runPredictionScaleMultiEtaDiagnostics( ...
    eventRoot,samplingMode,diagnosticEveryNValidEvents,maxEventsPerPhase)
%RUNPREDICTIONSCALEMULTIETADIAGNOSTICS Offline paired eta diagnostics.
% This function consumes a completed all-event F050 record. It never calls
% BLCMAES, never modifies an archived main-run MAT file, and never lets a
% diagnostic candidate enter a population, history archive, or CMA update.
%
% Examples
%   runPredictionScaleMultiEtaDiagnostics
%   runPredictionScaleMultiEtaDiagnostics([], 'all', 1, 3)
%   runPredictionScaleMultiEtaDiagnostics([], 'interval', 5, 3)
%   runPredictionScaleMultiEtaDiagnostics([], 'uniform_per_phase', 1, 3)
%
% Modes:
%   all               every valid recorded state
%   interval          every Nth valid state in a run
%   uniform_per_phase up to M evenly spaced states in each run and phase

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(eventRoot)
    eventRoot = fullfile(projectRoot,'results_prediction_scale_all_events_2D', ...
        'SMD1_SMD3_SMD6_SMD8_10runs');
end
if nargin < 2 || isempty(samplingMode), samplingMode = 'uniform_per_phase'; end
if nargin < 3 || isempty(diagnosticEveryNValidEvents)
    diagnosticEveryNValidEvents = 1;
end
if nargin < 4 || isempty(maxEventsPerPhase), maxEventsPerPhase = 3; end
samplingMode = normaliseSamplingMode(samplingMode);
validateattributes(diagnosticEveryNValidEvents,{'numeric'}, ...
    {'scalar','integer','positive'});
validateattributes(maxEventsPerPhase,{'numeric'}, ...
    {'scalar','integer','positive'});

stateFile = fullfile(eventRoot,'all_event_states.mat');
mainRoot = fullfile(eventRoot,'main_F050_runs');
assert(exist(stateFile,'file') == 2, ...
    'Missing all-event state archive: %s',stateFile);
assert(exist(fullfile(mainRoot,'injection_runs.csv'),'file') == 2, ...
    'Missing completed F050 run table: %s',mainRoot);
loadedStates = load(stateFile,'snapshots','eventTable','actualCandidateTable', ...
    'coverageTable','config');
snapshots = loadedStates.snapshots;
assert(~isempty(snapshots), ...
    'The all-event archive has no valid prediction auxiliary-sampling states.');
mainRuns = readtable(fullfile(mainRoot,'injection_runs.csv'));
mainRuns = mainRuns(mainRuns.prediction_inject_count == 3,:);

diagnosticRoot = fullfile(eventRoot,'multiscale_diagnostics', ...
    diagnosticDirectoryName(samplingMode,diagnosticEveryNValidEvents, ...
        maxEventsPerPhase));
if exist(diagnosticRoot,'dir') ~= 7, mkdir(diagnosticRoot); end
config = makeConfig(eventRoot,samplingMode,diagnosticEveryNValidEvents, ...
    maxEventsPerPhase,loadedStates.config);
saveOrValidateConfig(diagnosticRoot,config);

candidateFile = fullfile(diagnosticRoot,'multiscale_candidates.csv');
candidateTable = loadCandidateTable(candidateFile);
selectionRows = repmat(emptySelectionRow(),0,1);
selectedCount = 0;
global ulFunctionEvaluations llFunctionEvaluations;
savedUFE = ulFunctionEvaluations;
savedLFE = llFunctionEvaluations;
savedRng = rng;

for smdNo = unique([snapshots.SMD])
    smdSnapshots = snapshots([snapshots.SMD] == smdNo);
    for runNo = unique([smdSnapshots.run_no])
        runSnapshots = smdSnapshots([smdSnapshots.run_no] == runNo);
        unsortedStates = [runSnapshots.event];
        [~,order] = sort([unsortedStates.event_ordinal]);
        runSnapshots = runSnapshots(order);
        states = [runSnapshots.event];
        choice = selectPredictionScaleDiagnosticEvents(states,samplingMode, ...
            diagnosticEveryNValidEvents,maxEventsPerPhase);
        for stateIndex = 1:numel(states)
            state = states(stateIndex);
            selectionRows(end+1,1) = makeSelectionRow(runSnapshots(stateIndex), ...
                state,choice(stateIndex)); %#ok<AGROW>
            if ~choice(stateIndex).selected, continue; end
            selectedCount = selectedCount+1;
            eventId = stateIdentifier(runSnapshots(stateIndex).SMD, ...
                runSnapshots(stateIndex).run_no,state);
            existing = matchingEventRows(candidateTable,eventId);
            if ~isempty(candidateTable) && sum(existing) == 8
                fprintf('Reuse paired diagnostics %s\n',eventId);
                continue;
            end
            if ~isempty(candidateTable) && any(existing)
                candidateTable(existing,:) = [];
            end
            BI = loadRunBI(mainRuns,mainRoot,runSnapshots(stateIndex));
            [rows,totals] = evaluatePredictionScaleDiagnosticState(state,BI, ...
                runSnapshots(stateIndex).SMD,runSnapshots(stateIndex).run_no, ...
                runSnapshots(stateIndex).seed,config.etas);
            assert(numel(rows) == 8 && all([rows.raw_upper_matches_logged_main] ...
                | abs([rows.eta]-state.eta) > 1e-12), ...
                'The recovered eta=0.5 sample did not reproduce its logged raw upper point.');
            newTable = struct2table(rows);
            if isempty(candidateTable), candidateTable = newTable;
            else, candidateTable = [candidateTable;newTable]; end %#ok<AGROW>
            candidateTable = sortrows(candidateTable, ...
                {'SMD','run_no','generation','eta','candidate_index'});
            writetable(candidateTable,candidateFile);
            fprintf(['Paired eta diagnostic %s: %d candidates, UFE=%d, LFE=%d, ' ...
                'direction residual %.3g\n'],eventId,totals.candidate_count, ...
                totals.UFE,totals.LFE, ...
                totals.direction_reconstruction_relative_residual);
        end
    end
end

assert(isequal(savedRng,rng), ...
    'Offline diagnostic changed MATLAB global RNG state.');
assert(isequaln(ulFunctionEvaluations,savedUFE) && ...
    isequaln(llFunctionEvaluations,savedLFE), ...
    'Offline diagnostic leaked UFE/LFE into the main counters.');
selectionTable = struct2table(selectionRows);
selectionTable = sortrows(selectionTable,{'SMD','run_no','generation'});
writetable(selectionTable,fullfile(diagnosticRoot,'multiscale_selection.csv'));
assert(selectedCount > 0 && ~isempty(candidateTable), ...
    'No state was selected for the multi-eta diagnostic.');
selectedRows = selectionTable(selectionTable.selected ~= 0,:);
for selectedIndex = 1:height(selectedRows)
    eventId = tableText(selectedRows.state_id,selectedIndex);
    assert(sum(matchingEventRows(candidateTable,eventId)) == 8, ...
        'A selected state does not have exactly 4 eta x 2 candidate rows.');
end

[runSummary,stageSummary,pairedTests] = ...
    summarisePredictionScaleStageDiagnostic(candidateTable);
writetable(runSummary,fullfile(diagnosticRoot,'multiscale_run_summary.csv'));
writetable(stageSummary,fullfile(diagnosticRoot,'multiscale_stage_summary.csv'));
writetable(pairedTests,fullfile(diagnosticRoot,'multiscale_paired_tests.csv'));
figureFiles = plotPredictionScaleStageDiagnostic(stageSummary, ...
    fullfile(diagnosticRoot,'figures'));
writeReport(fullfile(diagnosticRoot,'multiscale_report.md'),config, ...
    selectionTable,stageSummary);
experiment = struct('eventRoot',eventRoot,'diagnosticRoot',diagnosticRoot, ...
    'config',config,'selection',selectionTable,'candidates',candidateTable, ...
    'runSummary',runSummary,'stageSummary',stageSummary, ...
    'pairedTests',pairedTests,'figureFiles',{figureFiles}, ...
    'notes',['All four eta values are re-evaluated with the same recovered ' ...
        'logged upper directions and the same private lower stream per ' ...
        'candidate index. The eta=0.5 row is deliberately re-evaluated so ' ...
        'the four-scale comparison has identical isolated lower randomness.']);
save(fullfile(diagnosticRoot,'multiscale_experiment.mat'),'experiment','-v7');
fprintf('\nCompleted paired multi-eta diagnostics: %s\n',diagnosticRoot);
end

function BI = loadRunBI(mainRuns,mainRoot,snapshot)
match = mainRuns.SMD == snapshot.SMD & mainRuns.run_no == snapshot.run_no;
assert(sum(match) == 1, ...
    'Cannot locate one F050 run MAT file for SMD%d run %d.', ...
    snapshot.SMD,snapshot.run_no);
row = mainRuns(find(match,1),:);
matFile = fullfile(mainRoot,portablePath(tableText(row.mat_file,1)));
loaded = load(matFile,'BI');
BI = loaded.BI;
assert(BI.u_dim == 2 && BI.l_dim == 3, ...
    'The multi-eta diagnostic currently requires DU=2, DL=3.');
end

function row = makeSelectionRow(snapshot,state,choice)
row = emptySelectionRow();
row.SMD = snapshot.SMD; row.run_no = snapshot.run_no; row.seed = snapshot.seed;
row.state_id = stateIdentifier(snapshot.SMD,snapshot.run_no,state);
row.stage = state.stage; row.generation = state.generation;
row.event_ordinal = state.event_ordinal; row.progress = state.progress;
row.selected = double(choice.selected); row.reason = choice.reason;
row.sampling_mode = choice.sampling_mode;
row.stage_rank = choice.stage_rank; row.stage_event_count = choice.stage_event_count;
end

function row = emptySelectionRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'state_id','', ...
    'stage','','generation',NaN,'event_ordinal',NaN,'progress',NaN, ...
    'selected',NaN,'reason','','sampling_mode','', ...
    'stage_rank',NaN,'stage_event_count',NaN);
end

function tableData = loadCandidateTable(fileName)
if exist(fileName,'file') == 2
    tableData = readtable(fileName);
else
    tableData = table();
end
end

function mask = matchingEventRows(tableData,eventId)
if isempty(tableData), mask = false(0,1); return; end
if iscell(tableData.state_id)
    mask = strcmp(tableData.state_id,eventId);
else
    mask = strcmp(cellstr(tableData.state_id),eventId);
end
end

function config = makeConfig(eventRoot,mode,everyN,maxPerPhase,mainConfig)
config = struct('version','paired_multi_eta_from_all_event_states_v2', ...
    'eventRoot',eventRoot,'samplingMode',mode, ...
    'diagnosticEveryNValidEvents',everyN, ...
    'maxEventsPerPhase',maxPerPhase,'etas',[0.25 0.5 0.75 1.0], ...
    'directionSource','recovered_logged_f050_raw_offsets', ...
    'lowerEvaluation',['isolated private stream reset by event and ' ...
        'candidate index across eta'], ...
    'eta05Treatment',['eta=0.5 has the same logged raw upper position but ' ...
        'is re-evaluated under the isolated lower stream for fairness'], ...
    'mainConfigurationVersion',mainConfig.version);
end

function saveOrValidateConfig(outputRoot,config)
fileName = fullfile(outputRoot,'multiscale_config.mat');
if exist(fileName,'file') ~= 2
    save(fileName,'config');
    return;
end
loaded = load(fileName,'config');
assert(isequaln(loaded.config,config), ...
    'This multi-eta output directory contains a different configuration.');
end

function writeReport(fileName,config,selection,stageSummary)
file = fopen(fileName,'w');
assert(file >= 0,'Cannot write multi-eta diagnostic report: %s',fileName);
cleaner = onCleanup(@() fclose(file)); %#ok<NASGU>
fprintf(file,'# Paired multi-eta diagnostic\n\n');
fprintf(file,'Sampling mode: `%s`. Selected %d of %d valid events.\n\n', ...
    config.samplingMode,sum(selection.selected),height(selection));
fprintf(file,['Each selected state uses its two logged F050 upper directions ' ...
    'at eta = 0.25, 0.50, 0.75, and 1.00. All four rows are evaluated ' ...
    'under a private, paired lower-search stream.\n\n']);
fprintf(file,'## Selection coverage\n\n');
for smdNo = unique(selection.SMD)'
    for stageCell = unique(selection(selection.SMD == smdNo,:).stage)'
        stage = tableText(stageCell,1);
        subset = selection(selection.SMD == smdNo & ...
            strcmp(selection.stage,stage),:);
        fprintf(file,'- SMD%d %s: %d/%d states selected.\n',smdNo,stage, ...
            sum(subset.selected),height(subset));
    end
end
fprintf(file,'\n## Cross-run stage summaries\n\n');
for smdNo = unique(stageSummary.SMD)'
    subset = stageSummary(stageSummary.SMD == smdNo,:);
    for rowIndex = 1:height(subset)
        row = subset(rowIndex,:);
        fprintf(file,['- SMD%d %s eta=%.2f: n=%d runs; Q_rank=%.3f; ' ...
            'Q_pred=%.3f; median Delta UF=%.4g.\n'], ...
            smdNo,tableText(row.stage,1),row.eta,row.independent_runs, ...
            row.Q_rank_median,row.Q_pred_median,row.delta_UF_median);
    end
end
end

function directory = diagnosticDirectoryName(mode,everyN,maxPerPhase)
switch mode
    case 'all'
        directory = 'all_valid_states';
    case 'interval'
        directory = sprintf('every_%d_valid_events',everyN);
    case 'uniform_per_phase'
        directory = sprintf('uniform_%d_per_phase_per_run',maxPerPhase);
    otherwise
        error('BLCMAES:DiagnosticSamplingMode','Unknown sampling mode.');
end
end

function mode = normaliseSamplingMode(value)
mode = lower(char(value));
allowed = {'all','interval','uniform_per_phase'};
assert(ismember(mode,allowed), ...
    'samplingMode must be ''all'', ''interval'', or ''uniform_per_phase''.');
end

function id = stateIdentifier(smdNo,runNo,state)
id = sprintf('SMD%d_run%02d_gen%04d',smdNo,runNo,state.generation);
end

function path = portablePath(path)
path = strrep(path,'/',filesep); path = strrep(path,'\',filesep);
end

function text = tableText(column,index)
if iscell(column), text = column{index}; else, text = char(column(index)); end
end
