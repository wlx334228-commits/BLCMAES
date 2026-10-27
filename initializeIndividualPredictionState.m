function state = initializeIndividualPredictionState(BI,options)
%INITIALIZEINDIVIDUALPREDICTIONSTATE Create the no-future-information state.
% A4 uses its own reproducible stream so choosing two transformed members
% never changes the global RNG sequence used by CMA-ES or lower searches.
if nargin < 2 || isempty(options)
    options = individualPredictionCorrectionOptions(BI);
end
pending = struct('valid',false,'x_pred',NaN(1,BI.u_dim), ...
    'source_generation',NaN,'model',struct());
state = struct('history',zeros(0,5),'pending',pending, ...
    'log',struct([]),'bestUF',-Inf,'partialSelectionStream',[]);
if strcmp(options.mode,'A4')
    state.partialSelectionStream = RandStream('mt19937ar', ...
        'Seed',options.partialSelectionSeed);
end
end
