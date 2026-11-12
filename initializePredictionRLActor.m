function rl = initializePredictionRLActor(BI,options)
%INITIALIZEPREDICTIONRLACTOR Private per-run REINFORCE actor state.
% The actor is deliberately independent from CMA-ES: it receives only a
% compact pre-sampling state and controls the auxiliary distribution only.
% Its private RandStream avoids changing the global random sequence used by
% CMA-ES and by all lower-level searches.

assert(BI.u_dim == 2, ...
    'BLCMAES:PredictionRL','The initial RL actor is defined for DU=2.');
stateDim = 10;
hidden = options.rlHiddenUnits;
stream = RandStream('mt19937ar','Seed',options.rlRandomSeed);
actor = struct();
actor.W1 = 0.10 * randn(stream,hidden,stateDim);
actor.b1 = zeros(hidden,1);
actor.W2 = 0.05 * randn(stream,4,hidden);
actor.b2 = zeros(4,1);
etaFraction = (options.rlFixedEta-options.rlEtaMin) / ...
    (options.rlEtaMax-options.rlEtaMin);
etaFraction = min(max(etaFraction,1e-6),1-1e-6);
actor.b2(1) = log(etaFraction/(1-etaFraction));
actor.b2(2) = 0; % delta=0 is the original CMA-ES shape.
actor.b2(3:4) = log(options.rlInitialActionStd);
actor.step = 0;
actor.mW1 = zeros(size(actor.W1)); actor.vW1 = zeros(size(actor.W1));
actor.mb1 = zeros(size(actor.b1)); actor.vb1 = zeros(size(actor.b1));
actor.mW2 = zeros(size(actor.W2)); actor.vW2 = zeros(size(actor.W2));
actor.mb2 = zeros(size(actor.b2)); actor.vb2 = zeros(size(actor.b2));

rl = struct('enabled',true,'mode',options.rlMode,'actor',actor, ...
    'actorStream',stream,'stateCount',0,'stateMean',zeros(stateDim,1), ...
    'stateM2',zeros(stateDim,1),'previousEta',options.rlFixedEta, ...
    'previousDelta',0,'previousReward',0,'previousDirectPairRate',0.5, ...
    'previousAuxDirectionPerformance',[0,0], ...
    'rewardBaseline',0,'rewardBaselineInitialized',false, ...
    'pendingDecision',emptyPredictionRLDecision(BI.u_dim), ...
    'lastUpdate',emptyPredictionRLUpdate(BI.u_dim));
end
