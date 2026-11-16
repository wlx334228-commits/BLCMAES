function experiment = runSMDPredictionScaleStageDiagnostic( ...
    outputRoot,numberOfRuns,problemNumbers,diagnosticEveryNValidEvents, ...
    enableMultiScaleDiagnostic,diagnosticSamplingMode,maxEventsPerPhase)
%RUNSMDPREDICTIONSCALESTAGEDIAGNOSTIC All-event auxiliary-scale analysis.
% Main F050 runs remain unchanged: 5 normal + 1 direct prediction + two
% eta=0.5 auxiliary candidates. Every real valid auxiliary-sampling event
% is copied after the run ends. Phase labels use each completed run's own
% [UFE_start,UFE_end] interval and are never available to the optimizer.
%
% Cost-controlled, evenly distributed diagnostic states in every phase:
%   runSMDPredictionScaleStageDiagnostic([],10,[1 3 6 8],1,true, ...
%       'uniform_per_phase',3)
%
% Diagnose every valid state (this can be expensive):
%   runSMDPredictionScaleStageDiagnostic([],10,[1 3 6 8],1,true,'all',3)
%
% Record actual F050 events only, with no extra diagnostic evaluations:
%   runSMDPredictionScaleStageDiagnostic([],10,[1 3 6 8],1,false)

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 2 || isempty(numberOfRuns), numberOfRuns = 10; end
if nargin < 3 || isempty(problemNumbers), problemNumbers = [1 3 6 8]; end
if nargin < 4 || isempty(diagnosticEveryNValidEvents)
    diagnosticEveryNValidEvents = 1;
end
if nargin < 5 || isempty(enableMultiScaleDiagnostic)
    enableMultiScaleDiagnostic = true;
end
if nargin < 6 || isempty(diagnosticSamplingMode)
    diagnosticSamplingMode = 'uniform_per_phase';
end
if nargin < 7 || isempty(maxEventsPerPhase), maxEventsPerPhase = 3; end
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'},{'vector','integer','>=',1,'<=',8});
validateattributes(diagnosticEveryNValidEvents,{'numeric'}, ...
    {'scalar','integer','positive'});
validateattributes(enableMultiScaleDiagnostic,{'logical','numeric'}, ...
    {'scalar','binary'});
validateattributes(maxEventsPerPhase,{'numeric'}, ...
    {'scalar','integer','positive'});
enableMultiScaleDiagnostic = logical(enableMultiScaleDiagnostic);
diagnosticSamplingMode = normaliseSamplingMode(diagnosticSamplingMode);
assert(all(ismember(problemNumbers,[1 3 6 8])) && ...
    numel(unique(problemNumbers)) == numel(problemNumbers), ...
    'This first all-event module supports unique SMD1, SMD3, SMD6, SMD8 only.');
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_prediction_scale_all_events_2D', ...
        sprintf('%s_%druns_multiscale_v2',problemTag(problemNumbers),numberOfRuns));
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

config = makeConfig(numberOfRuns,problemNumbers,diagnosticEveryNValidEvents, ...
    enableMultiScaleDiagnostic,diagnosticSamplingMode,maxEventsPerPhase);
saveOrValidateConfig(outputRoot,config);
smokeRoot = fullfile(outputRoot,'smoke_test');
if exist(fullfile(smokeRoot,'prediction_scale_all_event_test_report.mat'),'file') ~= 2
    testPredictionScaleAllEventDiagnostic(smokeRoot);
else
    fprintf('Reuse completed all-event diagnostic smoke test: %s\n',smokeRoot);
end

% The main F050 optimizer finishes before any diagnostic candidate is made.
mainRoot = fullfile(outputRoot,'main_F050_runs');
runSMDPredictionInjection(mainRoot,numberOfRuns,problemNumbers,3, ...
    'direct_plus_auxiliary',5,40,'cma_gaussian',0.5,'quadratic');
mainRuns = readtable(fullfile(mainRoot,'injection_runs.csv'));
mainRuns = mainRuns(mainRuns.prediction_inject_count == 3 & ...
    ismember(mainRuns.SMD,problemNumbers),:);
assert(height(mainRuns) == numel(problemNumbers)*numberOfRuns, ...
    'The F050 main-run table is incomplete.');
mainRuns = sortrows(mainRuns,{'SMD','run_no'});

eventRows = repmat(emptyEventRow(),0,1);
actualCandidateRows = repmat(emptyActualCandidateRow(),0,1);
coverageRows = repmat(emptyCoverageRow(),0,1);
snapshots = repmat(struct('SMD',NaN,'run_no',NaN,'seed',NaN, ...
    'event',struct()),0,1);

for mainIndex = 1:height(mainRuns)
    mainRow = mainRuns(mainIndex,:);
    matFile = fullfile(mainRoot,portablePath(tableText(mainRow.mat_file,1)));
    loaded = load(matFile,'injectionLog','BI','ins');
    [states,runInfo] = collectPredictionScaleAllEventStates( ...
        loaded.injectionLog,loaded.BI,loaded.ins.UFEs);
    selection = selectPredictionScaleDiagnosticEvents(states, ...
        diagnosticSamplingMode,diagnosticEveryNValidEvents,maxEventsPerPhase);
    coverageRows(end+1,1) = makeCoverageRow(mainRow,runInfo); %#ok<AGROW>
    for stateIndex = 1:numel(states)
        state = states(stateIndex);
        multiScaleDiagnosed = enableMultiScaleDiagnostic && selection(stateIndex).selected;
        [eventRow,auxiliaryRows] = buildPredictionScaleActualEventRows( ...
            state,mainRow.SMD,mainRow.run_no,mainRow.seed,multiScaleDiagnosed);
        eventRows(end+1,1) = eventRow; %#ok<AGROW>
        actualCandidateRows = [actualCandidateRows; auxiliaryRows]; %#ok<AGROW>
        snapshots(end+1,1) = struct('SMD',mainRow.SMD,'run_no', ...
            mainRow.run_no,'seed',mainRow.seed,'event',state); %#ok<AGROW>
    end
end

eventTable = struct2table(eventRows);
actualCandidateTable = struct2table(actualCandidateRows);
coverageTable = struct2table(coverageRows);
assert(~isempty(eventTable), ...
    ['No valid prediction-point auxiliary-sampling event was recorded. ' ...
     'Check the F050 prediction configuration before interpreting stage data.']);
eventTable = sortrows(eventTable,{'SMD','run_no','generation'});
actualCandidateTable = sortrows(actualCandidateTable, ...
    {'SMD','run_no','generation','auxiliary_index'});
writetable(eventTable,fullfile(outputRoot,'all_event_sampling_events.csv'));
writetable(actualCandidateTable, ...
    fullfile(outputRoot,'all_event_auxiliary_candidates.csv'));
writetable(coverageTable,fullfile(outputRoot,'all_event_phase_coverage.csv'));
save(fullfile(outputRoot,'all_event_states.mat'), ...
    'snapshots','eventTable','actualCandidateTable','coverageTable','config','-v7');

[actualRunSummary,actualStageSummary] = summarisePredictionScaleActualEvents( ...
    eventTable,actualCandidateTable);
writetable(actualRunSummary,fullfile(outputRoot,'all_event_actual_run_summary.csv'));
writetable(actualStageSummary,fullfile(outputRoot,'all_event_actual_stage_summary.csv'));
actualFigureRoot = fullfile(outputRoot,'figures_actual_events');
actualFigureFiles = plotPredictionScaleAllEventActual(eventTable, ...
    actualCandidateTable,actualFigureRoot);

diagnosticExperiment = struct();
diagnosticRunSummary = table(); diagnosticStageSummary = table();
diagnosticPaired = table(); diagnosticFigureFiles = cell(0,1);
if enableMultiScaleDiagnostic
    % The completed all-event state archive is now immutable.  The helper
    % below evaluates only isolated diagnostic candidates and reuses the
    % logged F050 directions; it does not rerun or alter the main optimizer.
    diagnosticExperiment = runPredictionScaleMultiEtaDiagnostics(outputRoot, ...
        diagnosticSamplingMode,diagnosticEveryNValidEvents,maxEventsPerPhase);
    diagnosticRunSummary = diagnosticExperiment.runSummary;
    diagnosticStageSummary = diagnosticExperiment.stageSummary;
    diagnosticPaired = diagnosticExperiment.pairedTests;
    diagnosticFigureFiles = diagnosticExperiment.figureFiles;
end
writeReport(fullfile(outputRoot,'all_event_scale_report.md'),config,coverageTable, ...
    actualStageSummary,diagnosticStageSummary);
experiment = struct('outputRoot',outputRoot,'mainRoot',mainRoot,'config',config, ...
    'mainRuns',mainRuns,'events',eventTable, ...
    'actualAuxiliaryCandidates',actualCandidateTable,'phaseCoverage',coverageTable, ...
    'actualRunSummary',actualRunSummary,'actualStageSummary',actualStageSummary, ...
    'diagnosticCandidates',diagnosticCandidates(diagnosticExperiment), ...
    'diagnosticRunSummary',diagnosticRunSummary, ...
    'diagnosticStageSummary',diagnosticStageSummary, ...
    'diagnosticPaired',diagnosticPaired, ...
    'diagnosticExperiment',diagnosticExperiment, ...
    'actualFigureFiles',{actualFigureFiles}, ...
    'diagnosticFigureFiles',{diagnosticFigureFiles}, ...
    'notes',['Actual event records consume no extra evaluations. Multi-scale ' ...
        'rows are offline, private-stream diagnostics with separately counted UFE/LFE.']);
save(fullfile(outputRoot,'all_event_scale_experiment.mat'),'experiment','-v7');
fprintf('\nAll-event prediction-scale analysis completed: %s\n',outputRoot);
end

function config = makeConfig(numberOfRuns,problemNumbers,frequency,enabled, ...
        samplingMode,maxPerPhase)
config = struct('version','all_events_postrun_phase_v2','totalDim',5, ...
    'upperDim',2,'lowerDim',3,'problems',problemNumbers, ...
    'numberOfRuns',numberOfRuns,'mainMode','F050', ...
    'normalCandidateCount',5,'directCandidateCount',1, ...
    'auxiliaryCandidateCount',2,'mainAuxiliaryScale',0.5, ...
    'predictionSampleCount',40,'historyGenerations',10, ...
    'predictionStartGeneration',10,'enableMultiScaleDiagnostic',enabled, ...
    'diagnosticEveryNValidEvents',frequency, ...
    'diagnosticSamplingMode',samplingMode, ...
    'maxEventsPerPhase',maxPerPhase,'etas',[0.25 0.5 0.75 1.0], ...
    'phaseDefinition',['post-run p=(UFE_after_event-UFE_start)/' ...
        '(UFE_end-UFE_start): early [0,1/3), middle [1/3,2/3), late [2/3,1]'], ...
    'randomness',['same logged F050 z1,z2 per event across eta; lower ' ...
        'stream reset by event and candidate index across eta']);
end

function saveOrValidateConfig(outputRoot,config)
fileName = fullfile(outputRoot,'all_event_scale_config.mat');
if exist(fileName,'file') ~= 2
    save(fileName,'config');
    return;
end
loaded = load(fileName,'config');
assert(isequaln(loaded.config,config), ...
    ['This output folder has a different all-event configuration. Use a new ' ...
     'directory to avoid combining incompatible records.']);
end

function tableData = loadDiagnosticTable(fileName)
if exist(fileName,'file') == 2
    tableData = readtable(fileName);
else
    tableData = table();
end
end

function value = diagnosticCandidates(experiment)
if isempty(fieldnames(experiment))
    value = table();
else
    value = experiment.candidates;
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

function row = makeCoverageRow(mainRow,runInfo)
row = struct('SMD',mainRow.SMD,'run_no',mainRow.run_no,'seed',mainRow.seed, ...
    'main_UFE_end',runInfo.UFE_end,'valid_event_count',runInfo.valid_event_count, ...
    'UFE_start',runInfo.UFE_start,'UFE_end',runInfo.UFE_end, ...
    'UFE_interval',runInfo.UFE_interval, ...
    'phase_assignment_available',double(runInfo.phase_assignment_available), ...
    'early_event_count',runInfo.early_event_count, ...
    'middle_event_count',runInfo.middle_event_count, ...
    'late_event_count',runInfo.late_event_count, ...
    'unclassified_event_count',runInfo.unclassified_event_count);
end

function row = emptyEventRow()
[row,~] = buildPredictionScaleActualEventRows(emptyStateForColumns(),1,1,1,false);
end

function row = emptyActualCandidateRow()
[~,rows] = buildPredictionScaleActualEventRows(emptyStateForColumns(),1,1,1,false);
row = rows(1);
end

function row = emptyCoverageRow()
row = struct('SMD',NaN,'run_no',NaN,'seed',NaN,'main_UFE_end',NaN, ...
    'valid_event_count',NaN,'UFE_start',NaN,'UFE_end',NaN, ...
    'UFE_interval',NaN,'phase_assignment_available',NaN, ...
    'early_event_count',NaN,'middle_event_count',NaN,'late_event_count',NaN, ...
    'unclassified_event_count',NaN);
end

function state = emptyStateForColumns()
state = struct('x_pred',[0 0],'C',eye(5),'sigma',1,'eta',0.5, ...
    'event_ordinal',NaN,'stage','','generation',NaN,'log_index',NaN, ...
    'progress',NaN,'UFE_start',NaN,'UFE_end',NaN, ...
    'UFE_before_sampling',NaN,'LFE_before_sampling',NaN, ...
    'UFE_after_population',NaN,'LFE_after_population',NaN, ...
    'cma_mean',zeros(1,5),'direct_UF',NaN,'normal_UF',zeros(1,5), ...
    'auxiliary_indices',[6 7],'auxiliary_joint',zeros(2,5), ...
    'auxiliary_UF',[0;0],'auxiliary_raw_joint',zeros(2,5), ...
    'auxiliary_sampled_joint',zeros(2,5), ...
    'auxiliary_boundary_repaired',false(2,5),'mean_update_applied',false, ...
    'population_joint',zeros(8,5),'population_UF',zeros(8,1), ...
    'selected_parent_indices',zeros(1,0), ...
    'history_best_UF_before_sampling',-Inf);
end

function writeReport(fileName,config,coverage,actualSummary,diagnosticSummary)
file = fopen(fileName,'w');
assert(file >= 0,'Cannot write all-event scale report: %s',fileName);
cleaner = onCleanup(@() fclose(file)); %#ok<NASGU>
fprintf(file,'# All-event prediction-point auxiliary scale analysis\n\n');
fprintf(file,['Main F050 policy: %d normal + %d direct + %d auxiliary at ' ...
    'eta=%.2f. Every valid realised auxiliary event is recorded.\n\n'], ...
    config.normalCandidateCount,config.directCandidateCount, ...
    config.auxiliaryCandidateCount,config.mainAuxiliaryScale);
fprintf(file,'## Per-run phase coverage\n\n');
for smdNo = unique(coverage.SMD)'
    rows = coverage(coverage.SMD == smdNo,:);
    fprintf(file,['- SMD%d: %d runs; median valid events %.1f; total event ' ...
        'counts early/middle/late = %d/%d/%d.\n'],smdNo,height(rows), ...
        median(rows.valid_event_count),sum(rows.early_event_count), ...
        sum(rows.middle_event_count),sum(rows.late_event_count));
end
fprintf(file,'\n## Actual F050 event summaries\n\n');
for smdNo = unique(actualSummary.SMD)'
    rows = actualSummary(actualSummary.SMD == smdNo,:);
    for stageCell = unique(rows.stage)'
        stage = tableText(stageCell,1);
        item = rows(strcmp(rows.stage,stage),:);
        fprintf(file,['- SMD%d %s: n=%d independent runs; median sigma %.4g; ' ...
            'median auxiliary RMS scale %.4g; Q_rank %.3f; Q_pred %.3f.\n'], ...
            smdNo,stage,item.independent_runs,item.sigma_median, ...
            item.auxiliary_rms_scale_median,item.Q_rank_median,item.Q_pred_median);
    end
end
if ~isempty(diagnosticSummary)
    fprintf(file,'\n## Offline multi-scale diagnostic\n\n');
    fprintf(file,['Eta values are compared only after aggregation within ' ...
        'each run and phase. Selection mode: %s; interval N=%d; ' ...
        'uniform phase cap=%d.\n'],config.diagnosticSamplingMode, ...
        config.diagnosticEveryNValidEvents,config.maxEventsPerPhase);
else
    fprintf(file,'\nOffline multi-scale diagnostic was disabled.\n');
end
end

function path = portablePath(path)
path = strrep(path,'/',filesep); path = strrep(path,'\',filesep);
end

function text = tableText(column,index)
if iscell(column), text = column{index}; else, text = char(column(index)); end
end

function tag = problemTag(problemNumbers)
tag = sprintf('SMD%d_',problemNumbers);
tag = tag(1:end-1);
end
