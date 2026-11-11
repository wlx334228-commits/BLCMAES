function model = fitLocalUpperPredictionModel( ...
    history,meanBefore,generation,BI,options,modelStream)
%FITLOCALUPPERPREDICTIONMODEL Select the configured local prediction model.
% This dispatcher deliberately affects only the construction of x_pred.
% The archive, injection, evaluation, selection, and CMA-ES update paths
% remain exactly those used by the existing prediction-injection code.

if nargin < 6, modelStream = []; end
switch options.predictionModelType
    case 'quadratic'
        model = fitLocalUpperQuadratic(history,meanBefore,generation,options);
    case 'neural_mlp'
        model = fitLocalUpperNeuralMLP(history,meanBefore,generation, ...
            BI,options,modelStream);
    otherwise
        error('BLCMAES:PredictionModel', ...
            'Unsupported predictionModelType: %s',options.predictionModelType);
end
end
