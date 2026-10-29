function state = initializePredictionInjectionState(BI,options)
%INITIALIZEPREDICTIONINJECTIONSTATE Build archive and private auxiliary RNG.
if nargin < 2 || isempty(options)
    options = predictionInjectionOptions(BI);
end
pending = struct('valid',false,'x_pred',NaN(1,BI.u_dim), ...
    'source_generation',NaN,'model',struct());
state = struct('history',zeros(0,5),'pending',pending, ...
    'log',struct([]),'bestUF',-Inf,'auxiliaryStream',[]);
if options.usesAuxiliary
    % A private stream preserves the original global random sequence used by
    % the main CMA-ES and every lower-level search.  All K>=2 auxiliary
    % candidates draw consecutively from this one stream.
    state.auxiliaryStream = RandStream('mt19937ar', ...
        'Seed',options.auxiliaryRandomSeed);
end
end
