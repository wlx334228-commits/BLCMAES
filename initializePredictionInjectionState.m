function state = initializePredictionInjectionState(BI,options)
%INITIALIZEPREDICTIONINJECTIONSTATE Build archive and private auxiliary RNG.
if nargin < 2 || isempty(options)
    options = predictionInjectionOptions(BI);
end
pending = struct('valid',false,'x_pred',NaN(1,BI.u_dim), ...
    'source_generation',NaN,'model',struct());
% Archive layout is [generation, individual_index, xu(1:DU), UF].  It is
% deliberately dimension-dependent so the same injection code can retain
% true evaluated upper samples for DU=2, DU=5, and DU=10.
state = struct('history',zeros(0,BI.u_dim+3),'pending',pending, ...
    'log',struct([]),'bestUF',-Inf,'auxiliaryStream',[], ...
    'modelStream',[],'rlState',[]);
if options.usesAuxiliary
    % A private stream preserves the original global random sequence used by
    % the main CMA-ES and every lower-level search.  In the original direct
    % policy, K>=2 needs this stream; in the auxiliary-only policy, every
    % K>=1 does.  All auxiliary candidates draw consecutively from it.
    state.auxiliaryStream = RandStream('mt19937ar', ...
        'Seed',options.auxiliaryRandomSeed);
end
if options.usesPrediction && strcmp(options.predictionModelType,'neural_mlp')
    % Online MLP initialisation must not consume the global stream used by
    % the original CMA-ES or lower-level optimisers.  This persistent
    % private stream is used only to initialise each locally re-trained MLP;
    % no training information is shared across independent runs.
    state.modelStream = RandStream('mt19937ar', ...
        'Seed',options.neuralRandomSeed);
end
if options.rlEnabled
    % One actor is initialised per independent BL-CMA-ES run.  It is never
    % shared across runs and its private stream cannot perturb CMA-ES RNG.
    state.rlState = initializePredictionRLActor(BI,options);
end
end
