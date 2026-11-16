function candidates = generatePredictionScaleDiagnosticCandidates(state,etas,z)
%GENERATEPREDICTIONSCALEDIAGNOSTICCANDIDATES Paired upper-only samples.
% Every eta uses the same two rows of z.  This function is deterministic,
% performs no objective evaluation, and does not access MATLAB's global RNG.

validateattributes(etas,{'numeric'},{'vector','real','finite','positive'});
assert(size(z,1) == 2 && size(z,2) == numel(state.x_pred), ...
    'z must contain exactly two standard-normal vectors in upper dimension.');
upperDim = numel(state.x_pred);
Cuu = (state.C(1:upperDim,1:upperDim) + ...
    state.C(1:upperDim,1:upperDim)')/2;
[B,values] = eig(Cuu);
eigenvalues = real(diag(values));
assert(all(isfinite(eigenvalues)) && all(eigenvalues > 0), ...
    'The recorded upper CMA covariance is not positive definite.');
D = sqrt(eigenvalues)';
factor = B*diag(D);
cmaRmsScale = state.sigma*sqrt(trace(Cuu)/upperDim);
template = struct('eta',NaN,'candidate_index',NaN,'z',NaN(1,upperDim), ...
    'raw_upper',NaN(1,upperDim),'actual_upper',NaN(1,upperDim), ...
    'boundary_repaired',false(1,upperDim),'raw_offset',NaN(1,upperDim), ...
    'actual_offset',NaN(1,upperDim),'raw_distance',NaN, ...
    'actual_distance',NaN,'relative_cma_distance',NaN, ...
    'standardized_distance',NaN,'upper_covariance',NaN(upperDim), ...
    'upper_factor',factor,'Cuu_eigenvalues',eigenvalues);
candidates = repmat(template,numel(etas)*2,1);
row = 0;
for etaIndex = 1:numel(etas)
    eta = etas(etaIndex);
    for candidateIndex = 1:2
        row = row+1;
        rawOffset = eta*state.sigma*(z(candidateIndex,:)*factor');
        rawUpper = state.x_pred+rawOffset;
        [actualUpper,repaired] = repairUpper(rawUpper,state.x_pred, ...
            state.u_lb,state.u_ub);
        actualOffset = actualUpper-state.x_pred;
        whitened = (actualOffset/(eta*state.sigma))/factor';
        candidates(row).eta = eta;
        candidates(row).candidate_index = candidateIndex;
        candidates(row).z = z(candidateIndex,:);
        candidates(row).raw_upper = rawUpper;
        candidates(row).actual_upper = actualUpper;
        candidates(row).boundary_repaired = repaired;
        candidates(row).raw_offset = rawOffset;
        candidates(row).actual_offset = actualOffset;
        candidates(row).raw_distance = norm(rawOffset);
        candidates(row).actual_distance = norm(actualOffset);
        candidates(row).relative_cma_distance = actualOffsetDistance(actualOffset, ...
            cmaRmsScale);
        candidates(row).standardized_distance = norm(whitened);
        candidates(row).upper_covariance = eta^2*state.sigma^2*Cuu;
    end
end
end

function [actual,repaired] = repairUpper(raw,reference,lower,upper)
actual = raw;
high = actual > upper;
low = actual < lower;
actual(high) = (reference(high)+upper(high))/2;
actual(low) = (reference(low)+lower(low))/2;
repaired = high | low;
end

function value = actualOffsetDistance(offset,rmsScale)
if rmsScale <= 0 || ~isfinite(rmsScale)
    value = NaN;
else
    value = norm(offset)/rmsScale;
end
end
