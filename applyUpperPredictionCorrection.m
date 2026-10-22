function [CMA,state] = applyUpperPredictionCorrection(CMA,meanBefore,POP, ...
    generation,BI,options,state,updateApplied,generationBestUF,eliteUF,UFE,LFE)
%APPLYUPPERPREDICTIONCORRECTION Blend the normal and predicted upper means.
% Call AFTER the original complete updateCMAES; pc/ps/C/sigma/B/D/invsqrtC
% remain exactly as calculated by that update. This is an external translation
% of the sampling centre, not an extra selected step in the evolution paths.
% At the next iteration updateCMAES's xold is this translated centre.
if isempty(state)
    state = struct('history',zeros(0,5),'log',struct([]),'bestUF',-Inf);
end
populationUF = [POP.UF];
populationUFMin = min(populationUF);
populationUFMedian = median(populationUF);
populationUFStd = std(populationUF,0);
currentBestUF = max([state.bestUF,populationUF,eliteUF]);
% A correction at t affects the already-evaluated population at t+1.  This
% update is logging only: no future value is available to the correction at t.
if ~isempty(state.log)
    previous = state.log(end);
    state.log(end).next_generation = generation;
    state.log(end).next_generation_best_UF = generationBestUF;
    state.log(end).next_population_UF_min = populationUFMin;
    state.log(end).next_population_UF_median = populationUFMedian;
    state.log(end).next_population_UF_std = populationUFStd;
    state.log(end).next_elite_UF = eliteUF;
    state.log(end).next_best_so_far_UF = currentBestUF;
    state.log(end).next_generation_best_UF_delta = ...
        generationBestUF - previous.generation_best_UF;
    state.log(end).next_population_UF_median_delta = ...
        populationUFMedian - previous.population_UF_median;
    state.log(end).next_elite_UF_delta = eliteUF - previous.elite_UF;
    state.log(end).next_best_so_far_UF_delta = ...
        currentBestUF - previous.best_so_far_UF;
    state.log(end).next_UFE = UFE;
    state.log(end).next_LFE_total = LFE;
    state.log(end).next_UFE_delta = UFE - previous.UFE;
    state.log(end).next_LFE_delta = LFE - previous.LFE_total;
end
if options.enabled
    rows = zeros(numel(POP),5);
    for i = 1:numel(POP)
        rows(i,:) = [generation,i,POP(i).UX,POP(i).UF];
    end
    state.history = [state.history;rows];
    state.history = state.history(state.history(:,1) >= ...
        generation-options.historyGenerations+1,:);
end
state.bestUF = currentBestUF;
entry = struct('generation',generation,'m_before',meanBefore, ...
    'm_cma',NaN(1,2),'x_pred',NaN(1,2),'d_cma',NaN(1,2), ...
    'd_pred',NaN(1,2),'d_new',NaN(1,2),'m_new',meanBefore, ...
    'd_actual',zeros(1,2),'valid_fit',false,'correction_applied',false, ...
    'weight_requested',options.weight,'weight_used',0, ...
    'normal_weight_requested',1-options.weight,'normal_weight_used',1, ...
    'prediction_weight_requested',options.weight, ...
    'prediction_weight_used',0,'mean_combination','endpoint_blend', ...
    'correction_requested',zeros(1,2),'correction_limited',zeros(1,2), ...
    'correction_actual',zeros(1,2),'correction_norm',0, ...
    'correction_limit',Inf,'cap_applied',false,'boundary_repaired',false, ...
    'boundary_adjustment',zeros(1,2),'mean_update_applied',updateApplied, ...
    'correction_scheduled',false, ...
    'population_UF_min',populationUFMin, ...
    'population_UF_median',populationUFMedian, ...
    'population_UF_std',populationUFStd, ...
    'next_generation',NaN,'next_generation_best_UF',NaN, ...
    'next_population_UF_min',NaN,'next_population_UF_median',NaN, ...
    'next_population_UF_std',NaN,'next_elite_UF',NaN, ...
    'next_best_so_far_UF',NaN,'next_generation_best_UF_delta',NaN, ...
    'next_population_UF_median_delta',NaN,'next_elite_UF_delta',NaN, ...
    'next_best_so_far_UF_delta',NaN,'next_UFE',NaN, ...
    'next_LFE_total',NaN,'next_UFE_delta',NaN,'next_LFE_delta',NaN, ...
    'normal_move_norm',NaN,'predicted_move_norm',NaN, ...
    'requested_correction_norm',0,'correction_to_cma_ratio',NaN, ...
    'requested_correction_to_cma_ratio',NaN, ...
    'upper_sigma_radius',NaN,'correction_to_sigma_ratio',NaN, ...
    'requested_correction_to_sigma_ratio',NaN, ...
    'correction_mahalanobis',NaN, ...
    'requested_correction_mahalanobis',NaN, ...
    'cos_pred_cma',NaN,'cos_correction_pc',NaN, ...
    'cos_correction_ps',NaN,'pc_upper_norm',NaN,'ps_upper_norm',NaN, ...
    'upper_covariance_condition',NaN, ...
    'upper_lower_covariance_norm',NaN, ...
    'upper_lower_correlation_norm',NaN, ...
    'conditional_lower_shift_requested',NaN(1,BI.l_dim), ...
    'conditional_lower_shift_requested_norm',NaN, ...
    'conditional_lower_shift_actual',NaN(1,BI.l_dim), ...
    'conditional_lower_shift_actual_norm',NaN, ...
    'fit_train_RMSE',NaN,'fit_train_R2',NaN, ...
    'fit_LOOCV_RMSE',NaN,'fit_LOOCV_R2',NaN, ...
    'selected_spread_eigenvalues',NaN(1,2), ...
    'selected_spread_condition',NaN,'hessian_eigenvalues',NaN(1,2), ...
    'hessian_rcond',NaN,'design_condition',NaN,'z_pred_norm',NaN, ...
    'prediction_to_nearest_sample',NaN, ...
    'prediction_to_nearest_sample_normalized',NaN, ...
    'prediction_in_hull',NaN, ...
    'generation_best_UF',generationBestUF,'best_so_far_UF',state.bestUF, ...
    'elite_UF',eliteUF,'UFE',UFE,'LFE_total',LFE, ...
    'sigma_after',CMA.sigma,'status','terminal_no_update','model',struct());
entry = appendCMAStateDiagnostics(entry,CMA,BI);
if updateApplied
    entry.m_cma = CMA.xmean(1:2);
    entry.m_new = entry.m_cma;
    entry.d_cma = entry.m_cma - meanBefore;
    entry.d_new = entry.d_cma;
    entry.d_actual = entry.d_cma;
    entry.normal_move_norm = norm(entry.d_cma);
    entry.status = 'disabled';
    entry.correction_scheduled = options.enabled && ...
        generation >= options.startGeneration && ...
        mod(generation-options.startGeneration,options.correctionInterval) == 0;
    if entry.correction_scheduled
        model = fitLocalUpperQuadratic(state.history,meanBefore,generation,options);
        entry.model = model;
        entry = appendModelDiagnostics(entry,model);
        entry.valid_fit = model.valid;
        entry.status = model.reason;
        if model.valid
            entry.x_pred = model.x_pred;
            entry.d_pred = model.x_pred - meanBefore;
            % Both inputs are points in the same decision space.  The
            % correction relative to the normal mean is therefore
            %   alpha * (x_pred - m_cma),
            % which gives m_new=(1-alpha)*m_cma+alpha*x_pred.  This is
            % deliberately different from the former additive rule
            % m_cma+alpha*(x_pred-m_before): the normal CMA step is also
            % given its requested (1-alpha) coefficient.
            correction = options.weight * (entry.x_pred - entry.m_cma);
            entry.correction_requested = correction;
            entry.predicted_move_norm = norm(entry.d_pred);
            entry.requested_correction_norm = norm(correction);
            entry.cos_pred_cma = safeCosine(entry.d_pred,entry.d_cma);
            entry = appendRequestedCorrectionDiagnostics(entry,correction,CMA,BI);
            weight = options.weight;
            if options.limitCorrection
                % RMS radius of the updated upper marginal distribution.
                limit = options.maxCorrectionSigmaFactor * CMA.sigma * ...
                    sqrt(max(0,trace(CMA.C(1:2,1:2))));
                entry.correction_limit = limit;
                if isfinite(norm(correction)) && norm(correction) > limit
                    weight = weight * limit / norm(correction);
                    correction = weight * (entry.x_pred - entry.m_cma);
                    entry.cap_applied = true;
                end
            end
            candidate = entry.m_cma + correction;
            if all(isfinite(candidate)) && all(isfinite(correction))
                entry.weight_used = weight;
                entry.normal_weight_used = 1-weight;
                entry.prediction_weight_used = weight;
                entry.correction_limited = correction;
                entry.d_new = candidate - meanBefore;
                % The existing sampler repairs an offending coordinate to
                % the midpoint of the reference centre and that bound.
                % Here the reference is the normal, uncorrected CMA mean.
                anchor = min(max(entry.m_cma,BI.u_lb),BI.u_ub);
                high = candidate > BI.u_ub;
                low = candidate < BI.u_lb;
                candidate(high) = (anchor(high)+BI.u_ub(high))/2;
                candidate(low) = (anchor(low)+BI.u_lb(low))/2;
                entry.boundary_repaired = any(high | low);
                entry.boundary_adjustment = candidate - ...
                    (entry.m_cma + correction);
                CMA.xmean(1:2) = candidate;
                entry.m_new = candidate;
                entry.d_actual = candidate - meanBefore;
                entry.correction_actual = candidate - entry.m_cma;
                entry.correction_norm = norm(entry.correction_actual);
                entry = appendActualCorrectionDiagnostics(entry,CMA,BI);
                entry.correction_applied = entry.correction_norm > 0;
                entry.status = 'applied';
                if ~entry.correction_applied
                    entry.status = 'zero_correction';
                end
            else
                entry.status = 'invalid_correction';
            end
        end
    elseif options.enabled
        % Preserve a row for every generation, while fitting and correcting
        % only on the configured schedule (for example 10, 15, 20, ...).
        entry.status = 'not_scheduled';
    end
end
if isempty(state.log)
    state.log = entry;
else
    state.log(end+1,1) = entry;
end
end

function entry = appendCMAStateDiagnostics(entry,CMA,BI)
Cuu = CMA.C(1:2,1:2);
Cul = CMA.C(1:2,BI.u_dim+1:end);
entry.upper_sigma_radius = CMA.sigma * sqrt(max(0,trace(Cuu)));
entry.upper_covariance_condition = cond(Cuu);
entry.pc_upper_norm = norm(CMA.pc(1:2));
entry.ps_upper_norm = norm(CMA.ps(1:2));
entry.upper_lower_covariance_norm = norm(Cul,'fro');
upperStd = sqrt(max(real(diag(Cuu)),eps));
lowerStd = sqrt(max(real(diag(CMA.C(BI.u_dim+1:end, ...
    BI.u_dim+1:end))),eps));
upperLowerCorrelation = bsxfun(@rdivide, ...
    bsxfun(@rdivide,Cul,upperStd),lowerStd');
entry.upper_lower_correlation_norm = norm(upperLowerCorrelation,'fro');
end

function entry = appendModelDiagnostics(entry,model)
entry.fit_train_RMSE = model.fitTrainRMSE;
entry.fit_train_R2 = model.fitTrainR2;
entry.fit_LOOCV_RMSE = model.fitLOOCVRMSE;
entry.fit_LOOCV_R2 = model.fitLOOCVR2;
entry.selected_spread_eigenvalues = model.selectedSpreadEigenvalues(:)';
entry.selected_spread_condition = model.selectedSpreadCondition;
entry.hessian_eigenvalues = model.hessianEigenvalues(:)';
entry.hessian_rcond = model.hessianRcond;
entry.design_condition = model.designCondition;
entry.z_pred_norm = model.zPredNorm;
entry.prediction_to_nearest_sample = model.predictionToNearestSample;
entry.prediction_to_nearest_sample_normalized = ...
    model.predictionToNearestSampleNormalized;
entry.prediction_in_hull = model.predictionInHull;
end

function entry = appendRequestedCorrectionDiagnostics(entry,correction,CMA,BI)
entry.requested_correction_to_cma_ratio = norm(correction) / ...
    (entry.normal_move_norm + 1e-12);
entry.requested_correction_to_sigma_ratio = norm(correction) / ...
    (entry.upper_sigma_radius + 1e-12);
entry.requested_correction_mahalanobis = ...
    correctionMahalanobis(correction,CMA.C(1:2,1:2),CMA.sigma);
entry.conditional_lower_shift_requested = conditionalLowerShift( ...
    correction,CMA.C,BI.u_dim);
entry.conditional_lower_shift_requested_norm = ...
    norm(entry.conditional_lower_shift_requested);
end

function entry = appendActualCorrectionDiagnostics(entry,CMA,BI)
correction = entry.correction_actual;
entry.correction_to_cma_ratio = norm(correction) / ...
    (entry.normal_move_norm + 1e-12);
entry.correction_to_sigma_ratio = norm(correction) / ...
    (entry.upper_sigma_radius + 1e-12);
entry.correction_mahalanobis = ...
    correctionMahalanobis(correction,CMA.C(1:2,1:2),CMA.sigma);
entry.cos_correction_pc = safeCosine(correction,CMA.pc(1:2));
entry.cos_correction_ps = safeCosine(correction,CMA.ps(1:2));
entry.conditional_lower_shift_actual = conditionalLowerShift( ...
    correction,CMA.C,BI.u_dim);
entry.conditional_lower_shift_actual_norm = ...
    norm(entry.conditional_lower_shift_actual);
end

function value = correctionMahalanobis(correction,Cuu,sigma)
value = NaN;
if all(isfinite(correction)) && isfinite(sigma) && sigma > 0 && ...
        rcond(Cuu) > eps
    squaredDistance = correction * (Cuu \ correction');
    if isfinite(squaredDistance)
        value = sqrt(max(0,real(squaredDistance))) / sigma;
    end
end
end

function lowerShift = conditionalLowerShift(correction,C,upperDim)
lowerShift = NaN(1,size(C,1)-upperDim);
Cuu = C(1:upperDim,1:upperDim);
if all(isfinite(correction)) && rcond(Cuu) > eps
    lowerShift = (C(upperDim+1:end,1:upperDim) * ...
        (Cuu \ correction'))';
end
end

function value = safeCosine(a,b)
denominator = norm(a) * norm(b);
if isfinite(denominator) && denominator > 0
    value = dot(a,b) / denominator;
else
    value = NaN;
end
end
