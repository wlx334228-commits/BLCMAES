function manifest = runSMDGenerationDiagnostics(outputRoot)
%RUNSMDGENERATIONDIAGNOSTICS Run and log 10 BL-CMA-ES trials per SMD setting.
%   This experiment driver runs the unconstrained SMD1--SMD8 set at (2+3).
%   It only enables BLCMAES's read-only generationDiagnostic logger; it does
%   not add objective evaluations or alter the optimizer's search procedure.

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputRoot)
    % Keep the earlier generation-summary logs intact.  This expanded schema
    % needs a fresh run because old CSV files do not contain member-level data.
    outputRoot = fullfile(projectRoot,'results_population_detail');
end

addpath(fullfile(projectRoot,'BLEA_D'));

% getBLOPinfo's third argument is total dimension, not upper dimension.
% Keep only (2+3), SMD1--SMD8: 8 problems x 10 runs = 80 trials.
dimensionConfigs = struct('totalDim',5, 'uDim',2, 'lDim',3);
problemNumbers = 1:8;
numberOfRuns = 10;
numberOfExperiments = length(problemNumbers) * length(dimensionConfigs) * ...
                      numberOfRuns;
manifestColumns = {'SMD','u_dim','l_dim','run_no','seed', ...
                   'final_generation','final_UFE','final_LFE_total', ...
                   'final_generation_best_F','final_best_so_far_F', ...
                   'final_UF','final_LF', ...
                   'run_time_seconds','csv_file','mat_file', ...
                   'population_csv_file'};
manifestRows = cell(numberOfExperiments,length(manifestColumns));
manifestIndex = 0;

if ~exist(outputRoot,'dir')
    mkdir(outputRoot);
end

for configIndex = 1:length(dimensionConfigs)
    config = dimensionConfigs(configIndex);
    for smdNo = problemNumbers
        baseBI = getBLOPinfo('SMD',smdNo,config.totalDim);
        if baseBI.u_dim ~= config.uDim || baseBI.l_dim ~= config.lDim
            error('Unexpected SMD dimension mapping for total dimension %d.', ...
                  config.totalDim);
        end
        baseBI = applyOriginalExperimentBudget(baseBI);
        dimensionLabel = sprintf('DU%d_DL%d',baseBI.u_dim,baseBI.l_dim);
        outputDir = fullfile(outputRoot,sprintf('SMD%d',smdNo),dimensionLabel);
        if ~exist(outputDir,'dir')
            mkdir(outputDir);
        end

        for runNo = 1:numberOfRuns
            fileStem = sprintf('SMD%d_%s_run%02d',smdNo,dimensionLabel,runNo);
            csvFile = fullfile(outputDir,[fileStem '.csv']);
            matFile = fullfile(outputDir,[fileStem '.mat']);
            populationCsvFile = fullfile(outputDir,[fileStem '_population.csv']);
            ensureRunFilesDoNotExist(csvFile,matFile,populationCsvFile);

            % There are 80 unique deterministic seeds, one per independent
            % trial.  rng only sets the state; it consumes no random values.
            seed = (smdNo - 1) * numberOfRuns + runNo;
            rng(seed,'twister');
            rngStateBefore = rng;

            BI = baseBI;
            BI.generationDiagnostic = struct('enabled',true);
            tic;
            ins = BLCMAES(BI);
            runTimeSeconds = toc;

            generationLog = ins.generationLog;
            columns = ins.generationLogColumns;
            populationLog = ins.generationPopulationLog;
            populationColumns = ins.generationPopulationLogColumns;
            generationBestFColumn = find(strcmp(columns,'generation_best_F'),1);
            bestSoFarFColumn = find(strcmp(columns,'best_so_far_F'),1);
            if isempty(generationBestFColumn) || isempty(bestSoFarFColumn)
                error('Generation diagnostic does not provide both best-F columns.');
            end
            generationTable = array2table(generationLog,'VariableNames',columns);
            % Repeat run metadata in the CSV so it remains self-describing if
            % it is copied out of its results directory.
            rowCount = height(generationTable);
            generationTable.SMD = repmat(smdNo,rowCount,1);
            generationTable.u_dim = repmat(BI.u_dim,rowCount,1);
            generationTable.l_dim = repmat(BI.l_dim,rowCount,1);
            generationTable.run_no = repmat(runNo,rowCount,1);
            generationTable.seed = repmat(seed,rowCount,1);

            % One row is retained for every evaluated population member in a
            % generation.  Its xu1...xuDU and UF come from the same POP entry.
            populationTable = array2table(populationLog, ...
                'VariableNames',populationColumns);
            populationRowCount = height(populationTable);
            populationTable.SMD = repmat(smdNo,populationRowCount,1);
            populationTable.u_dim = repmat(BI.u_dim,populationRowCount,1);
            populationTable.l_dim = repmat(BI.l_dim,populationRowCount,1);
            populationTable.run_no = repmat(runNo,populationRowCount,1);
            populationTable.seed = repmat(seed,populationRowCount,1);

            metadata = createRunMetadata(BI,smdNo,runNo,seed,rngStateBefore, ...
                                         config.totalDim,runTimeSeconds);
            finalResult = struct('UF',ins.UF,'LF',ins.LF, ...
                                 'UX',ins.UX,'LX',ins.LX, ...
                                 'UFEs',ins.UFEs,'LFEs',ins.LFEs, ...
                                 'runTimeSeconds',runTimeSeconds);
            generationLogState = ins.generationLogState;
            writetable(generationTable,csvFile);
            writetable(populationTable,populationCsvFile);
            save(matFile,'generationLog','columns','generationLogState', ...
                 'populationLog','populationColumns','metadata','finalResult');

            manifestIndex = manifestIndex + 1;
            manifestRows(manifestIndex,:) = {smdNo, BI.u_dim, BI.l_dim, ...
                runNo, seed, generationLog(end,1), ins.UFEs, ins.LFEs, ...
                generationLog(end,generationBestFColumn), ...
                generationLog(end,bestSoFarFColumn), ins.UF, ins.LF, ...
                runTimeSeconds, ...
                csvFile, matFile, populationCsvFile};
            fprintf(['SMD%d (%d+%d) run %02d/%02d: seed=%d, gen=%d, ' ...
                     'generation_best_F=%g, best_so_far_F=%g, ' ...
                     'UFEs=%d, LFEs=%d, population_rows=%d, time=%.2fs\n'], ...
                    smdNo,BI.u_dim,BI.l_dim,runNo,numberOfRuns,seed, ...
                    generationLog(end,1), ...
                    generationLog(end,generationBestFColumn), ...
                    generationLog(end,bestSoFarFColumn), ...
                    ins.UFEs,ins.LFEs,populationRowCount,runTimeSeconds);
        end
    end
end

manifest = cell2table(manifestRows,'VariableNames',manifestColumns);
manifestCsv = fullfile(outputRoot,'SMD_generation_log_manifest.csv');
manifestMat = fullfile(outputRoot,'SMD_generation_log_manifest.mat');
writetable(manifest,manifestCsv);
save(manifestMat,'manifest','manifestColumns','dimensionConfigs', ...
     'problemNumbers','numberOfRuns');
fprintf('Saved %d run logs under:\n  %s\n',height(manifest),outputRoot);


function BI = applyOriginalExperimentBudget(BI)
% These are exactly the existing BLEA_runExperiment.m SMD budget settings.
switch BI.dim
    case 5
        BI.UmaxFEs = 2500;
        BI.UmaxImprFEs = 350;
        BI.LmaxFEs = 250;
        BI.LmaxImprFEs = 25;
    case 10
        BI.UmaxFEs = 3500;
        BI.UmaxImprFEs = 500;
        BI.LmaxFEs = 350;
        BI.LmaxImprFEs = 35;
    case 20
        BI.UmaxFEs = 5000;
        BI.UmaxImprFEs = 750;
        BI.LmaxFEs = 500;
        BI.LmaxImprFEs = 50;
    otherwise
        error('No original budget is defined for total dimension %d.',BI.dim);
end
BI.u_N = 50;
BI.l_N = 50;


function ensureRunFilesDoNotExist(csvFile,matFile,populationCsvFile)
% A complete run has a summary CSV, MAT, and member-level population CSV.
% Refuse to overwrite any of them so code revisions cannot mix in one run.
hasCsv = exist(csvFile,'file') == 2;
hasMat = exist(matFile,'file') == 2;
hasPopulationCsv = exist(populationCsvFile,'file') == 2;
if hasCsv || hasMat || hasPopulationCsv
    error(['Run output already exists or is incomplete; it will not be ' ...
           'overwritten:\n  %s\n  %s\n  %s'], ...
          csvFile,matFile,populationCsvFile);
end


function metadata = createRunMetadata(BI,smdNo,runNo,seed,rngStateBefore,totalDim,runTimeSeconds)
metadata = struct();
metadata.algorithm = 'BLCMAES';
metadata.diagnosticSchemaVersion = 'generation_diagnostics_v3_population_detail';
metadata.problem = sprintf('SMD%d',smdNo);
metadata.runNo = runNo;
metadata.seed = seed;
metadata.rngGenerator = 'twister';
metadata.rngStateBefore = rngStateBefore;
metadata.totalDim = totalDim;
metadata.uDim = BI.u_dim;
metadata.lDim = BI.l_dim;
metadata.UmaxFEs = BI.UmaxFEs;
metadata.UmaxImprFEs = BI.UmaxImprFEs;
metadata.LmaxFEs = BI.LmaxFEs;
metadata.LmaxImprFEs = BI.LmaxImprFEs;
metadata.u_N = BI.u_N;
metadata.l_N = BI.l_N;
metadata.covarianceMatrixDimension = BI.dim;
% The benchmark interface does not expose y*(xu) for an arbitrary current
% upper vector, so a truthful per-generation LL error would require extra work.
% It is intentionally omitted rather than adding an evaluation or optimization.
metadata.generationBestLLErrorAvailable = false;
metadata.generationRowTiming = [ ...
    'For a nonterminal row, population metrics describe generation t and ' ...
    'CMA state is recorded after that population updates CMA-ES.  The final ' ...
    'terminal row has mean_update_applied=0 and contains its pre-update CMA state.'];
metadata.updateEliteStateDescription = [ ...
    'generationLogState.updateEliteIndices and updateEliteWeights align ' ...
    'with log rows; they are empty for generation 0 and a terminal row.'];
metadata.populationLogDescription = [ ...
    'populationLog has one row per final generation POP member, with its ' ...
    'sample-order individual_index, upper vector xu1...xuDU, and UF.'];
metadata.runTimeSeconds = runTimeSeconds;
