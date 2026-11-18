function [snapshot,valid] = captureContinuousScaleSnapshot( ...
    CMA,rawUpper,rawLower,generation,BI,injectionState,elite,record, ...
    previousPopulation,UFE,LFE,globalRng)
%CAPTURECONTINUOUSSCALESNAPSHOT Read-only pre-injection state capture.
% This helper deliberately performs no objective evaluation and makes no
% random draw.  It is called only after the ordinary normal population has
% been sampled and before the pending quadratic prediction is injected.

snapshot = emptySnapshot(BI);
valid = false;
if isempty(injectionState) || ~isfield(injectionState,'pending') || ...
        ~injectionState.pending.valid
    return;
end
model = injectionState.pending.model;
if ~isstruct(model) || ~isfield(model,'valid') || ~model.valid || ...
        ~isfield(model,'H') || ~isfield(model,'scale') || ...
        ~isfield(model,'selectedSamples')
    return;
end
Hnormalized = (model.H + model.H')/2;
if ~isequal(size(Hnormalized),[BI.u_dim BI.u_dim]) || ...
        any(~isfinite(Hnormalized(:))) || ~isfinite(model.scale) || ...
        model.scale <= 0
    return;
end
% The existing quadratic predictor accepts only strict maxima: -H is SPD.
[~,notPositive] = chol(-Hnormalized);
if notPositive || rcond(Hnormalized) <= 0
    return;
end
Cuu = (CMA.C(1:BI.u_dim,1:BI.u_dim) + ...
    CMA.C(1:BI.u_dim,1:BI.u_dim)')/2;
if any(~isfinite(Cuu(:))) || rcond(Cuu) <= sqrt(eps) || ...
        ~isfinite(CMA.sigma) || CMA.sigma <= 0
    return;
end

% Match applyPredictionInjection's direct-point repair before computing d.
directIndex = CMA.lambda - 3 + 1;
if directIndex < 1 || directIndex > size(rawUpper,1), return; end
xRequested = injectionState.pending.x_pred;
if ~isreal(xRequested) || numel(xRequested) ~= BI.u_dim || ...
        any(~isfinite(xRequested))
    return;
end
xPred = xRequested;
high = xPred > BI.u_ub;
low = xPred < BI.u_lb;
reference = rawUpper(directIndex,:);
xPred(high) = (reference(high) + BI.u_ub(high))/2;
xPred(low) = (reference(low) + BI.u_lb(low))/2;

offset = xPred - CMA.xmean(1:BI.u_dim);
mahalanobisSquared = offset * (Cuu \ offset');
if ~isfinite(mahalanobisSquared) || mahalanobisSquared < -sqrt(eps)
    return;
end
d = sqrt(max(mahalanobisSquared,0) / (BI.u_dim*CMA.sigma^2));
Hphysical = Hnormalized / model.scale^2;
selectedUF = model.selectedSamples(:,end);
sF = std(selectedUF,0);
if ~isfinite(sF), return; end
q = CMA.sigma^2 * trace((-Hphysical)*Cuu) / ...
    (2*(sF + eps(max(1,sF))));
if ~isfinite(d) || ~isfinite(q)
    return;
end

snapshot.version = 'continuous_scale_snapshot_v1';
snapshot.next_generation = generation;
snapshot.generation = generation;
snapshot.UFE_before_sampling = UFE;
snapshot.LFE_before_sampling = LFE;
snapshot.global_rng_after_normal_sampling = globalRng;
snapshot.BI = BI;
snapshot.CMA = CMA;
snapshot.elite = elite;
snapshot.record = record;
snapshot.previous_population = previousPopulation;
snapshot.injection_state = serialisePredictionInjectionStateForBranch(injectionState);
snapshot.raw_normal_upper = rawUpper;
snapshot.raw_normal_lower = rawLower;
snapshot.normal_joint = [rawUpper,rawLower];
snapshot.x_pred_requested = xRequested;
snapshot.x_pred_actual = xPred;
snapshot.prediction_repaired = high | low;
snapshot.model = model;
snapshot.H_normalized = Hnormalized;
snapshot.H_physical = Hphysical;
snapshot.Cuu = Cuu;
snapshot.sigma = CMA.sigma;
snapshot.cma_mean = CMA.xmean;
snapshot.sF = sF;
snapshot.d = d;
snapshot.q = q;
snapshot.model_scale = model.scale;
snapshot.model_design_condition = scalarModelField(model,'designCondition');
snapshot.model_hessian_rcond = scalarModelField(model,'hessianRcond');
snapshot.model_reason = textModelField(model,'reason');
snapshot.history_best_UF = injectionState.bestUF;
snapshot.pending_source_generation = injectionState.pending.source_generation;
snapshot.snapshot_reason = 'valid_quadratic_prediction';
valid = true;
end

function snapshot = emptySnapshot(BI)
snapshot = struct('version','','next_generation',NaN,'generation',NaN, ...
    'UFE_before_sampling',NaN,'LFE_before_sampling',NaN, ...
    'global_rng_after_normal_sampling',struct(),'BI',BI,'CMA',struct(), ...
    'elite',[],'record',[],'previous_population',[], ...
    'injection_state',struct(),'raw_normal_upper',NaN(0,BI.u_dim), ...
    'raw_normal_lower',NaN(0,BI.l_dim),'normal_joint',NaN(0,BI.dim), ...
    'x_pred_requested',NaN(1,BI.u_dim),'x_pred_actual',NaN(1,BI.u_dim), ...
    'prediction_repaired',false(1,BI.u_dim),'model',struct(), ...
    'H_normalized',NaN(BI.u_dim),'H_physical',NaN(BI.u_dim), ...
    'Cuu',NaN(BI.u_dim),'sigma',NaN,'cma_mean',NaN(1,BI.dim), ...
    'sF',NaN,'d',NaN,'q',NaN,'model_scale',NaN, ...
    'model_design_condition',NaN,'model_hessian_rcond',NaN, ...
    'model_reason','','history_best_UF',NaN, ...
    'pending_source_generation',NaN,'snapshot_reason','');
end

function value = scalarModelField(model,name)
value = NaN;
if isfield(model,name) && isnumeric(model.(name)) && ...
        isscalar(model.(name)) && isfinite(model.(name))
    value = model.(name);
end
end

function value = textModelField(model,name)
value = '';
if isfield(model,name) && (ischar(model.(name)) || isstring(model.(name)))
    value = char(model.(name));
end
end
