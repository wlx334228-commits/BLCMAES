function report = testNeuralPredictionModel(outputRoot)
%TESTNEURALPREDICTIONMODEL Fast, objective-free safety checks for NN model.
% This test does not call a SMD objective or a lower-level search.  It
% verifies archive-only training, bounded surrogate maximisation, direct
% injection, deterministic private-stream use, and invalid-model fallback.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(projectRoot,'results_neural_prediction_tests');
end
if exist(outputRoot,'dir') ~= 7, mkdir(outputRoot); end

BI = getBLOPinfo('SMD',1,5);
BI.predictionInjection = neuralSettings(41);
options = predictionInjectionOptions(BI);
meanBefore = [0,0];
history = deterministicHistory();

globalBefore = rng;
stream = RandStream('mt19937ar','Seed',options.neuralRandomSeed);
neural = fitLocalUpperNeuralMLP(history,meanBefore,10,BI,options,stream);
globalAfter = rng;
assert(isequal(globalBefore,globalAfter), ...
    'Neural fitting must not advance MATLAB''s global random stream.');
assert(neural.valid && strcmp(neural.modelType,'neural_mlp'), ...
    'The deterministic local neural model should be valid.');
assert(isequal(size(neural.W1),[6,2]) && isequal(size(neural.W2),[1,6]), ...
    'The first NN version must be 2-6-1.');
assert(all(neural.x_pred >= BI.u_lb-1e-10) && ...
    all(neural.x_pred <= BI.u_ub+1e-10), ...
    'Neural prediction escaped the upper bounds.');
assert(isfinite(neural.trainMSE) && isfinite(neural.validationMSE) && ...
    isfinite(neural.predictedUF), ...
    'Neural fit diagnostics are not finite.');

quadratic = fitLocalUpperQuadratic(history,meanBefore,10,options);
assert(quadratic.valid, ...
    'The shared synthetic archive should also yield a valid quadratic model.');
fixedHistoryDiagnostic = struct('quadratic_x_pred',quadratic.x_pred, ...
    'quadratic_predicted_UF',quadratic.predictedUF, ...
    'neural_x_pred',neural.x_pred, ...
    'neural_predicted_UF',neural.predictedUF, ...
    'prediction_distance',norm(quadratic.x_pred-neural.x_pred));

% K=1 has no auxiliary sampling, so it isolates whether a predicted point
% is inserted before evaluation without changing normal sampling.
state = initializePredictionInjectionState(BI,options);
state.history = history;
state.pending = struct('valid',true,'x_pred',neural.x_pred, ...
    'source_generation',10,'model',neural);
CMA = struct('xmean',[meanBefore,zeros(1,BI.l_dim)],'C',eye(BI.dim), ...
    'sigma',0.25,'B',eye(BI.dim),'D',ones(1,BI.dim));
rawUpper = repmat(meanBefore,8,1) + ...
    repmat(linspace(-0.2,0.2,8)',1,2);
rawLower = zeros(8,BI.l_dim);
[upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
    11,BI,options,state);
assert(entry.effective_injected_count == 1 && entry.direct_index == 8 && ...
    sum(strcmp(entry.source_labels,'normal')) == 7, ...
    'K=1 neural prediction must leave exactly seven normal candidates.');
assert(norm(upper(8,:)-entry.x_pred_actual) <= 1e-12 && ...
    isequal(lower,rawLower), ...
    'The direct neural prediction was not passed to the evaluation input.');

badHistory = history;
badHistory(:,end) = 1;
invalid = fitLocalUpperNeuralMLP(badHistory,meanBefore,10,BI,options,stream);
assert(~invalid.valid && strcmp(invalid.reason,'degenerate_target'), ...
    'A constant-UF archive must reject neural prediction safely.');
report = struct('passed',true,'neural_model',neural, ...
    'quadratic_model',quadratic, ...
    'fixed_history_diagnostic',fixedHistoryDiagnostic, ...
    'injection_entry',entry,'invalid_model',invalid, ...
    'notes',['No true upper objective, lower optimisation, or function ' ...
     'evaluation is performed by this test.']);
save(fullfile(outputRoot,'neural_prediction_model_test_report.mat'),'report','-v7');
fprintf('Neural prediction model test passed: %s\n',outputRoot);
end

function settings = neuralSettings(seed)
settings = struct('mode','K1','predictionInjectCount',1, ...
    'candidatePolicy','direct_plus_auxiliary','predictionModelType','neural_mlp', ...
    'startGeneration',10,'historyGenerations',10,'sampleCount',40, ...
    'auxiliaryRandomSeed',500000+seed,'neuralRandomSeed',700000+seed, ...
    'randomSeed',seed,'logEnabled',true);
end

function history = deterministicHistory()
% Exactly 40 old, finite rows from a strict concave quadratic, ensuring the
% legacy quadratic control and the bounded NN optimiser both have a valid
% fixed-history test state.
gridX = linspace(-1,1,8);
gridY = linspace(-0.9,0.9,5);
history = zeros(40,5);
index = 0;
for i = 1:numel(gridX)
    for j = 1:numel(gridY)
        index = index+1;
        x = [gridX(i),gridY(j)];
        uf = 4 - 2*(x(1)-0.25)^2 - 1.5*(x(2)+0.2)^2;
        history(index,:) = [10,index,x,uf];
    end
end
end
