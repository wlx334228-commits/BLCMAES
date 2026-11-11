function diagnostic = diagnoseQuadraticVsNeuralFixedHistory( ...
    history,meanBefore,generation,BI,neuralSeed)
%DIAGNOSEQUADRATICVSNEURALFIXEDHISTORY Compare two models on one archive.
% This is an offline, no-evaluation diagnostic: it fits both models to the
% same already-evaluated archive and reports their surrogate predictions.
% It does not call an upper objective or a lower-level optimiser.

if nargin < 5 || isempty(neuralSeed), neuralSeed = 271828; end
base = struct('mode','K1','predictionInjectCount',1, ...
    'candidatePolicy','direct_plus_auxiliary','startGeneration',10, ...
    'historyGenerations',10,'sampleCount',40,'logEnabled',true, ...
    'neuralRandomSeed',neuralSeed);
BI.predictionInjection = base;
quadraticOptions = predictionInjectionOptions(BI);
quadraticOptions.predictionModelType = 'quadratic';
neuralOptions = quadraticOptions;
neuralOptions.predictionModelType = 'neural_mlp';
quadratic = fitLocalUpperQuadratic(history,meanBefore,generation, ...
    quadraticOptions);
stream = RandStream('mt19937ar','Seed',neuralSeed);
neural = fitLocalUpperNeuralMLP(history,meanBefore,generation,BI, ...
    neuralOptions,stream);
diagnostic = struct('generation',generation,'mean',meanBefore, ...
    'quadratic',quadratic,'neural',neural, ...
    'both_valid',quadratic.valid && neural.valid, ...
    'prediction_distance',NaN);
if diagnostic.both_valid
    diagnostic.prediction_distance = norm(quadratic.x_pred-neural.x_pred);
end
end
