function diagnostics = diagnoseNeuralPredictionFixedStates( ...
    quadraticResultRoot,outputRoot)
%DIAGNOSENEURALPREDICTIONFIXEDSTATES No-evaluation Q-versus-NN diagnostic.
% For every valid quadratic fit already stored in a completed Q run, train a
% new NN on exactly that model's selected archive rows and use the same
% sampling mean/generation.  This compares prediction locations and
% surrogate stability on fixed historical states only.  It intentionally
% performs no true upper evaluation and no lower-level optimisation.

if nargin < 2 || isempty(outputRoot), outputRoot = quadraticResultRoot; end
runs = readtable(fullfile(quadraticResultRoot,'injection_runs.csv'));
records = cell(0,16);
names = {'SMD','run_no','seed','generation','quadratic_valid','neural_valid', ...
    'quadratic_predicted_UF','neural_predicted_UF','prediction_distance', ...
    'quadratic_z_norm','neural_z_norm','neural_train_MSE', ...
    'neural_validation_MSE','neural_training_epochs', ...
    'neural_optimizer_converged','neural_prediction_at_boundary'};
for rowIndex = 1:height(runs)
    runFile = fullfile(quadraticResultRoot,portablePath(tableText(runs.mat_file,rowIndex)));
    loaded = load(runFile,'injectionLog','BI');
    for g = 1:numel(loaded.injectionLog)
        entry = loaded.injectionLog(g);
        quadratic = entry.next_prediction_model;
        if ~isstruct(quadratic) || ~isfield(quadratic,'modelType') || ...
                ~strcmp(quadratic.modelType,'quadratic') || ...
                ~isfield(quadratic,'selectedSamples') || ...
                isempty(quadratic.selectedSamples)
            continue;
        end
        BI = loaded.BI;
        settings = struct('mode','K1','predictionInjectCount',1, ...
            'candidatePolicy','direct_plus_auxiliary', ...
            'predictionModelType','neural_mlp','logEnabled',true, ...
            'startGeneration',10,'historyGenerations',10,'sampleCount',40, ...
            'neuralRandomSeed',900000 + 10000*runs.SMD(rowIndex) + ...
             100*runs.run_no(rowIndex) + g);
        BI.predictionInjection = settings;
        options = predictionInjectionOptions(BI);
        meanBefore = entry.cma_mean_before_sampling(1:BI.u_dim);
        stream = RandStream('mt19937ar','Seed',options.neuralRandomSeed);
        neural = fitLocalUpperNeuralMLP(quadratic.selectedSamples, ...
            meanBefore,entry.generation,BI,options,stream);
        distance = NaN;
        if quadratic.valid && neural.valid
            distance = norm(quadratic.x_pred-neural.x_pred);
        end
        records(end+1,:) = {runs.SMD(rowIndex),runs.run_no(rowIndex), ...
            runs.seed(rowIndex),entry.generation,logical(quadratic.valid), ...
            logical(neural.valid),scalarField(quadratic,'predictedUF'), ...
            scalarField(neural,'predictedUF'),distance, ...
            scalarField(quadratic,'zPredNorm'),scalarField(neural,'zPredNorm'), ...
            scalarField(neural,'trainMSE'),scalarField(neural,'validationMSE'), ...
            scalarField(neural,'trainingEpochs'), ...
            scalarField(neural,'optimizerConverged'), ...
            scalarField(neural,'predictionAtSearchBoundary')}; %#ok<AGROW>
    end
end
diagnostics = cell2table(records,'VariableNames',names);
writetable(diagnostics,fullfile(outputRoot, ...
    'neural_fixed_history_model_diagnostic.csv'));
save(fullfile(outputRoot,'neural_fixed_history_model_diagnostic.mat'), ...
    'diagnostics','-v7');
end

function value = scalarField(item,name)
value = NaN;
if isstruct(item) && isfield(item,name) && ...
        (isnumeric(item.(name)) || islogical(item.(name))) && ...
        isscalar(item.(name)) && isfinite(item.(name))
    value = double(item.(name));
end
end

function path = portablePath(relativePath)
path = strrep(relativePath,'/',filesep);
path = strrep(path,'\\',filesep);
end

function value = tableText(column,index)
if iscell(column)
    value = char(column{index});
elseif ischar(column)
    value = strtrim(column(index,:));
else
    value = char(column(index));
end
end
