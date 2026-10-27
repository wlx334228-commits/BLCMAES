% @Author: hxy
% @Date:   2017-11-15 11:48:12
% @Last Modified by:   hxy
% @Last Modified time: 2018-03-06 20:00:16

% i --- injection correction
% e --- elite preservation
% c --- constrained handling
% s --- advanced stopping criteria

% 2017/11/15 modify some termination criteria

function ins = BLCMAES(BI)
% Prediction variants are intentionally mutually exclusive.  The injection
% variant evaluates external candidates but never directly changes CMA.xmean.
meanPredictionOptions = predictionCorrectionOptions(BI);
individualPredictionOptions = individualPredictionCorrectionOptions(BI);
injectionPredictionOptions = predictionInjectionOptions(BI);
enabledPredictionStrategies = [meanPredictionOptions.enabled, ...
    individualPredictionOptions.enabled,injectionPredictionOptions.enabled];
if sum(enabledPredictionStrategies) > 1
    error('BLCMAES:PredictionStrategyConflict', ...
        'Only one prediction strategy may be enabled in a run.');
end
if injectionPredictionOptions.enabled && ...
        (meanPredictionOptions.logEnabled || individualPredictionOptions.logEnabled)
    error('BLCMAES:PredictionStrategyConflict', ...
        ['Disable the old mean/individual prediction logs when using B1/B2; ' ...
         'they would describe a different candidate population.']);
end
predictionState = [];
individualPredictionState = [];
injectionPredictionState = [];
global ulFunctionEvaluations;
global llFunctionEvaluations;
ulFunctionEvaluations = 0;
llFunctionEvaluations = 0;

elite = [];
record = [];
CMA = initCMAES(BI);
% Optional mean-trajectory logging. This is read-only bookkeeping and is
% disabled unless BI.meanTrajectoryDiagnostic.enabled is explicitly true.
meanTrajectoryDiagnostic = initializeMeanTrajectoryDiagnostic(BI);
if meanTrajectoryDiagnostic.enabled
    meanTrajectoryDiagnostic = appendMeanTrajectoryDiagnostic( ...
        meanTrajectoryDiagnostic, 0, CMA);
end
% Optional per-generation experiment log. This is read-only bookkeeping and
% is disabled unless BI.generationDiagnostic.enabled is explicitly true.
generationDiagnostic = initializeGenerationDiagnostic(BI);
if generationDiagnostic.enabled
    generationDiagnostic = appendInitialGenerationDiagnostic( ...
        generationDiagnostic, CMA, ulFunctionEvaluations, llFunctionEvaluations);
end
maxIter = ceil(BI.UmaxFEs/CMA.lambda);
ImprIter = ceil(BI.UmaxImprFEs/CMA.lambda);
for iter = 1 : maxIter
    if meanPredictionOptions.logEnabled
        % The centre that generates population iter; never a future mean.
        predictionMeanBefore = CMA.xmean(1:BI.u_dim);
    end
    if individualPredictionOptions.logEnabled
        % This remains the exact normal CMA-ES sampling centre.  The v2
        % strategy never writes CMA.xmean at this or any later point.
        individualSamplingMean = CMA.xmean(1:BI.u_dim);
    end
    if injectionPredictionOptions.logEnabled
        injectionSamplingMean = CMA.xmean;
    end
    if generationDiagnostic.enabled
        % This snapshots the real lower-level counter before the original
        % generation work begins; it does not alter the counter or RNG state.
        LFEAtGenerationStart = llFunctionEvaluations;
    end
    % Normal joint-variable sampling.  Draw order and the original first
    % boundary repair are unchanged.  V2 touches only upper coordinates after
    % all lambda samples exist and before any true evaluation occurs.
    sampledUpper = zeros(CMA.lambda,BI.u_dim);
    sampledLower = zeros(CMA.lambda,BI.l_dim);
    for i = 1 : CMA.lambda
        U = CMA.xmean + CMA.sigma * randn(1,BI.dim) .* CMA.D * CMA.B';
        flag = U > BI.xrange(2,:);
        U(flag) = (CMA.xmean(flag) + BI.xrange(2,flag))/2;
        flag = U < BI.xrange(1,:);
        U(flag) = (CMA.xmean(flag) + BI.xrange(1,flag))/2;
        sampledUpper(i,:) = U(1:BI.u_dim);
        sampledLower(i,:) = U(BI.u_dim+1:end);
    end
    if individualPredictionOptions.logEnabled
        [sampledUpper,individualPredictionEntry,individualPredictionState] = ...
            applyIndividualPredictionCorrection(CMA,sampledUpper,sampledLower, ...
                iter,BI,individualPredictionOptions,individualPredictionState);
    end
    if injectionPredictionOptions.logEnabled
        [sampledUpper,sampledLower,injectionPredictionEntry, ...
            injectionPredictionState] = applyPredictionInjection( ...
                CMA,sampledUpper,sampledLower,iter,BI, ...
                injectionPredictionOptions,injectionPredictionState);
    end
    for i = 1 : CMA.lambda
        POP(i).UX = sampledUpper(i,:);
        POP(i).LX = sampledLower(i,:);
    end
    if individualPredictionOptions.logEnabled
        % This stores the exact joint coordinates that the following lower
        % search and upper evaluation receive.  It is a read-only audit.
        individualPredictionEntry = recordIndividualPredictionEvaluationInput( ...
            individualPredictionEntry,POP,individualPredictionOptions);
    end
    if injectionPredictionOptions.logEnabled
        injectionPredictionEntry = recordPredictionInjectionEvaluationInput( ...
            injectionPredictionEntry,POP,injectionPredictionOptions);
    end
    % lower level search
    for i = 1 : CMA.lambda
        [POP(i).LX,POP(i).LF,POP(i).LC,POP(i).RF] = lowerLevelSearch(POP(i).UX,CMA,BI);
        [POP(i).UF,POP(i).UC] = UL_evaluate(POP(i).UX,POP(i).LX,BI.fn);
        POP(i).UFEs = ulFunctionEvaluations;
        POP(i).LFEs = llFunctionEvaluations;
    end
    if individualPredictionOptions.logEnabled
        % This records the exact (UX,LX) pair after the existing lower search
        % and immediately after its normal upper evaluation.  It is read-only.
        individualPredictionEntry = recordIndividualPredictionTrueEvaluation( ...
            individualPredictionEntry,POP,individualPredictionOptions);
    end
    if injectionPredictionOptions.logEnabled
        injectionPredictionEntry = recordPredictionInjectionTrueEvaluation( ...
            injectionPredictionEntry,POP);
    end
    % fitness assignment && refinement
    POP = assignUpperFitness(POP,BI);

    % find current best and optimal solution
    RF_idx = find([POP.RF]);
    if isempty(RF_idx) RF_idx = 1:length(POP); end
    [~,bestIdx] = max([POP(RF_idx).fit]);
    bestIdx = RF_idx(bestIdx);
    bestIndv = POP(bestIdx);

    % replace the elite
    % `extraRefinedElite` is populated only when the original branch refines
    % an old elite outside POP and genuinely performs a new upper evaluation.
    % It is archived for diagnostics/prediction, but never inserted into this
    % generation's CMA-ES parent pool.
    extraRefinedElite = [];
    if upperLevelComparator(bestIndv,elite,BI) || rand > 0.5
        [bestIndv,~] = refine(bestIndv,CMA,BI);
    	POP(bestIdx) = bestIndv;
    	if upperLevelComparator(bestIndv,elite,BI)
    		elite = bestIndv;
    	end
    else
        [elite,refinedOldElite] = refine(elite,CMA,BI);
        if refinedOldElite
            extraRefinedElite = elite;
        end
    end
    % re-assign fitness
    POP = assignUpperFitness(POP,BI);
    if individualPredictionOptions.logEnabled
        % Store the actual evaluated (therefore possibly repaired) members.
        % Fitting is delayed to here and creates only the next generation's
        % pending prediction; it cannot leak future information backwards.
        [individualPredictionState,individualPredictionEntry] = ...
            finalizeIndividualPredictionCorrection( ...
                individualPredictionState,individualPredictionEntry,POP, ...
                individualSamplingMean,iter,BI,individualPredictionOptions, ...
                max([POP.UF]),elite.UF,ulFunctionEvaluations, ...
                llFunctionEvaluations);
    end
    if injectionPredictionOptions.logEnabled
        [injectionPredictionState,injectionPredictionEntry] = ...
            finalizePredictionInjection(injectionPredictionState, ...
                injectionPredictionEntry,POP,elite,extraRefinedElite, ...
                injectionSamplingMean,iter,BI, ...
                injectionPredictionOptions,ulFunctionEvaluations, ...
                llFunctionEvaluations);
    end
    if generationDiagnostic.enabled
        % Save the final, already-evaluated population that this generation
        % supplies to the original termination check and CMA-ES update.
        generationDiagnostic = appendGenerationPopulationDiagnostic( ...
            generationDiagnostic,iter,POP);
    end
    % [elite.UF elite.LF CMA.sigma]
%     [ CMA.xmean CMA.sigma]
    % [elite.UF elite.LF elite.UC elite.LC CMA.sigma ]

    % recording
    elite.UFEs = ulFunctionEvaluations;
    elite.LFEs = llFunctionEvaluations;
    record = [record;elite];

    %% termination check
    reachTheTarget_b = abs(bestIndv.UF-BI.u_fopt) < BI.u_ftol;
    reachTheTarget_e = abs(elite.UF-BI.u_fopt) < BI.u_ftol;
    reachTheMaxFEs = ulFunctionEvaluations >= BI.UmaxFEs;
    reachTheFlat = iter>ImprIter && abs(record(iter).UF-record(iter-ImprIter+1).UF)/(abs(record(1).UF)+abs(record(iter).UF)) < BI.u_ftol; 
    reachTheFlat = reachTheFlat && abs(record(iter).UF-record(iter-ImprIter+1).UF) < BI.u_ftol;
    
    if reachTheTarget_b || reachTheTarget_e || reachTheMaxFEs || reachTheFlat
        if reachTheTarget_b
            elite = bestIndv;
        end
        % The original code does not execute updateCMAES on a terminating
        % generation.  Log its real, pre-update mean and mark that fact.
        if generationDiagnostic.enabled
            generationDiagnostic = appendGenerationDiagnostic( ...
                generationDiagnostic, iter, CMA, POP, bestIndv, bestIdx, elite, BI, ...
                ulFunctionEvaluations, llFunctionEvaluations, ...
                LFEAtGenerationStart, false, []);
        end
        if meanPredictionOptions.logEnabled
            [CMA,predictionState] = applyUpperPredictionCorrection( ...
                CMA,predictionMeanBefore,POP,iter,BI,meanPredictionOptions, ...
                predictionState,false,bestIndv.UF,elite.UF, ...
                ulFunctionEvaluations,llFunctionEvaluations);
        end
        if individualPredictionOptions.logEnabled
            individualPredictionEntry.termination_reason = ...
                getTerminationReason(reachTheTarget_b,reachTheTarget_e, ...
                    reachTheMaxFEs,reachTheFlat);
            individualPredictionState = appendIndividualPredictionLog( ...
                individualPredictionState,individualPredictionEntry,CMA,false, ...
                [],individualPredictionOptions);
        end
        if injectionPredictionOptions.logEnabled
            injectionPredictionEntry.termination_reason = ...
                getTerminationReason(reachTheTarget_b,reachTheTarget_e, ...
                    reachTheMaxFEs,reachTheFlat);
            injectionPredictionState = appendPredictionInjectionLog( ...
                injectionPredictionState,injectionPredictionEntry,CMA,false, ...
                [],injectionPredictionOptions);
        end
        break;
    end
    if generationDiagnostic.enabled || individualPredictionOptions.logEnabled || ...
            injectionPredictionOptions.logEnabled
        % updateInfo is copied from the exact ranking already used by the
        % original CMA-ES update.  It is only returned for logging.
        [CMA,updateInfo] = updateCMAES(CMA,POP,BI);
    else
        CMA = updateCMAES(CMA,POP,BI);
    end
    if meanPredictionOptions.logEnabled
        % Original mean, paths, covariance, sigma and eigensystem are already
        % updated above. Only the upper sampling centre may change here.
        [CMA,predictionState] = applyUpperPredictionCorrection( ...
            CMA,predictionMeanBefore,POP,iter,BI,meanPredictionOptions, ...
            predictionState,true,bestIndv.UF,elite.UF, ...
            ulFunctionEvaluations,llFunctionEvaluations);
    end
    if individualPredictionOptions.logEnabled
        individualPredictionEntry.termination_reason = 'continue';
        individualPredictionState = appendIndividualPredictionLog( ...
            individualPredictionState,individualPredictionEntry,CMA,true, ...
            updateInfo,individualPredictionOptions);
    end
    if injectionPredictionOptions.logEnabled
        injectionPredictionEntry.termination_reason = 'continue';
        injectionPredictionState = appendPredictionInjectionLog( ...
            injectionPredictionState,injectionPredictionEntry,CMA,true, ...
            updateInfo,injectionPredictionOptions);
    end
    if meanTrajectoryDiagnostic.enabled
        meanTrajectoryDiagnostic = appendMeanTrajectoryDiagnostic( ...
            meanTrajectoryDiagnostic, iter, CMA);
    end
    if generationDiagnostic.enabled
        % POP and elite remain the already-evaluated state of this generation;
        % CMA contains the actual next sampling mean and updated sigma.
        generationDiagnostic = appendGenerationDiagnostic( ...
            generationDiagnostic, iter, CMA, POP, bestIndv, bestIdx, elite, BI, ...
            ulFunctionEvaluations, llFunctionEvaluations, ...
            LFEAtGenerationStart, true, updateInfo);
    end
end

ins.UF = elite.UF;
ins.LF = elite.LF;
ins.UX = elite.UX;
ins.LX = elite.LX;
ins.UFEs = ulFunctionEvaluations;
ins.LFEs = llFunctionEvaluations;
ins.record = record;
if meanPredictionOptions.logEnabled
    ins.predictionLog = predictionState.log;
    ins.predictionOptions = meanPredictionOptions;
end
if individualPredictionOptions.logEnabled
    ins.individualPredictionLog = individualPredictionState.log;
    ins.individualPredictionOptions = individualPredictionOptions;
end
if injectionPredictionOptions.logEnabled
    ins.predictionInjectionLog = injectionPredictionState.log;
    ins.predictionInjectionOptions = injectionPredictionOptions;
end
if meanTrajectoryDiagnostic.enabled
    ins.meanTrajectory = meanTrajectoryDiagnostic.rows;
    ins.meanTrajectoryColumns = meanTrajectoryDiagnostic.columns;
end
if generationDiagnostic.enabled
    ins.generationLog = generationDiagnostic.rows;
    ins.generationLogColumns = generationDiagnostic.columns;
    ins.generationLogState.covariances = generationDiagnostic.covariances;
    ins.generationLogState.updateEliteIndices = ...
        generationDiagnostic.updateEliteIndices;
    ins.generationLogState.updateEliteWeights = ...
        generationDiagnostic.updateEliteWeights;
    ins.generationLogState.generationBestIndices = ...
        generationDiagnostic.generationBestIndices;
    ins.generationPopulationLog = generationDiagnostic.populationRows;
    ins.generationPopulationLogColumns = generationDiagnostic.populationColumns;
end

function cfit_ = combineConstraintWithFitness(fit_,c)
if all(c==0)
	cfit_ = fit_;
else
    cfit_ = -c;
    isFeasible = c == 0;
    if any(isFeasible)
        cfit_(isFeasible) = fit_(isFeasible);
        cfit_(~isFeasible) = min(fit_(isFeasible)) - c(~isFeasible);
    end
end

function POP = assignLowerFitness(POP)
fit_ = combineConstraintWithFitness([POP.LF],[POP.LC]);
for i = 1 : length(POP)
    POP(i).fit = fit_(i);
end

function POP = assignUpperFitness(POP,BI)
CV = [POP.UC];
if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
    CV = CV + [POP.LC];
end
fit_ = combineConstraintWithFitness([POP.UF],CV);
for i = 1 : length(POP)
	POP(i).fit = fit_(i);
	% ????? how to handle RF
	% according to latest experimental results, incorporating RF here seems to be meaningless
end

function [Q,upperEvaluated] = refine(P,CMA,BI)
Q = P;
upperEvaluated = false;
[Q.LX,Q.LF,Q.LC,Q.RF] = lowerLevelSearch(Q.UX,CMA,BI);
if lowerLevelComparator(Q,P)
    Q.RF = max(Q.RF,P.RF);
    [Q.UF,Q.UC] = UL_evaluate(Q.UX,Q.LX,BI.fn);
	upperEvaluated = true;
else
	Q = P;
end

function isNoWorseThan = upperLevelComparator(P,Q,BI)
if isempty(Q)
    isNoWorseThan = true;
else
    tmp = assignUpperFitness([P Q],BI);
    isNoWorseThan = tmp(1).fit >= tmp(2).fit;
end

function isNoWorseThan = lowerLevelComparator(P,Q)
if isempty(Q)
    isNoWorseThan = true;
else
    tmp = assignLowerFitness([P Q]);
    isNoWorseThan = tmp(1).fit >= tmp(2).fit;
end

function [bestLX,bestLF,bestLC,bestRF] = lowerLevelSearch(xu,CMA,BI)
sigma0 = 1;
LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
LCMA.sigma = sigma0;
LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end) * CMA.sigma^2;
LCMA.pc = CMA.pc(BI.u_dim+1:end) * CMA.sigma;
LCMA.ps = zeros(1,BI.l_dim);
lambda = 4+floor(3*log(BI.l_dim));
mu = floor(lambda/2);
weights = log(mu+1/2)-log(1:mu);
weights = weights/sum(weights);
mueff=sum(weights)^2/sum(weights.^2);
cc = (4+mueff/BI.l_dim) / (BI.l_dim+4 + 2*mueff/BI.l_dim);
cs = (mueff+2) / (BI.l_dim+mueff+5);
c1 = 2 / ((BI.l_dim+1.3)^2+mueff);
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((BI.l_dim+2)^2+mueff));
damps = 1 + 2*max(0, sqrt((mueff-1)/(BI.l_dim+1))-1) + cs;
chiN=BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
[LCMA.B,LCMA.D] = eig(LCMA.C);
LCMA.D = sqrt(diag(LCMA.D))';
LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
cy = sqrt(BI.l_dim)+2*BI.l_dim/(BI.l_dim+2);
bestIndv = [];
bestRF = false;

maxIter = ceil(BI.LmaxFEs / lambda);
ImprIter = ceil(BI.LmaxImprFEs / lambda);
record = zeros(1,maxIter);
for iter = 1 : maxIter
	for i = 1 : lambda
	    Q(i).LX = LCMA.xmean + LCMA.sigma * randn(1,BI.l_dim) .* LCMA.D * LCMA.B';
	    flag = Q(i).LX > BI.l_ub;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_ub(flag))/2;
	    flag = Q(i).LX < BI.l_lb;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_lb(flag))/2;
	    [Q(i).LF,Q(i).LC] = LL_evaluate(xu,Q(i).LX,BI.fn);
	end
	Q = assignLowerFitness(Q);
	[~, Xindex] = sort(-[Q.fit]);
	xold = LCMA.xmean;
	X = cell2mat(arrayfun(@(q)q.LX,Q','UniformOutput',false));
	Y = bsxfun(@minus,X(Xindex(1:mu),:),xold) / LCMA.sigma;
	Y = bsxfun(@times,Y,min(1,cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
	delta_xmean = weights * Y;
	LCMA.xmean = LCMA.xmean + delta_xmean * LCMA.sigma;
	C_mu = Y' * diag(weights) * Y;
	LCMA.ps = (1-cs)*LCMA.ps + sqrt(cs*(2-cs)*mueff) * delta_xmean * LCMA.invsqrtC;
	LCMA.pc = (1-cc)*LCMA.pc + sqrt(cc*(2-cc)*mueff) * delta_xmean;
	LCMA.C = (1-c1-cmu) * LCMA.C + c1 * (LCMA.pc'*LCMA.pc) + cmu * C_mu; 
	delta_sigma = (cs/damps)*(norm(LCMA.ps)/chiN - 1);
	LCMA.sigma = LCMA.sigma * exp(min(CMA.delta_sigma_max,delta_sigma));
	LCMA.C = triu(LCMA.C) + triu(LCMA.C,1)';
	[LCMA.B,LCMA.D] = eig(LCMA.C);
	LCMA.D = sqrt(diag(LCMA.D))';
    LCMA = repairCMA(LCMA);
	LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';

	% elite-preservation right????
	if lowerLevelComparator(Q(Xindex(1)),bestIndv)
		bestIndv = Q(Xindex(1));
	end
    record(iter) = bestIndv.LF;

    if (iter>ImprIter && abs(record(iter)-record(iter-ImprIter+1))/(abs(record(1))+abs(record(iter))) < 1e-4) ...
		|| (iter > ImprIter && abs(record(iter) - record(iter-ImprIter+1)) < 10*BI.l_ftol) ...
        || LCMA.sigma / sigma0 < 1e-2 ...
        || LCMA.sigma / sigma0 > 1e2
    	bestRF = true;
        break; 
    end
end
bestLX = bestIndv.LX;
bestLF = bestIndv.LF;
bestLC = bestIndv.LC;

function CMA = initCMAES(BI)
CMA.lambda = 4+floor(3*log(BI.dim));
% CMA.sigma = min(BI.xrange(2,:)-BI.xrange(1,:))/2;
CMA.sigma = 0.3*median(BI.xrange(2,:)-BI.xrange(1,:));
CMA.mu = floor(CMA.lambda/2);
CMA.weights = log(CMA.mu+1/2)-log(1:CMA.mu);
CMA.weights = CMA.weights/sum(CMA.weights);
CMA.mueff=sum(CMA.weights)^2/sum(CMA.weights.^2);
CMA.cc = (4+CMA.mueff/BI.dim) / (BI.dim+4 + 2*CMA.mueff/BI.dim);
CMA.cs = (CMA.mueff+2) / (BI.dim+CMA.mueff+5);
CMA.c1 = 2 / ((BI.dim+1.3)^2+CMA.mueff);
CMA.cmu = min(1-CMA.c1, 2 * (CMA.mueff-2+1/CMA.mueff) / ((BI.dim+2)^2+CMA.mueff));
CMA.damps = 1 + 2*max(0, sqrt((CMA.mueff-1)/(BI.dim+1))-1) + CMA.cs;
CMA.chiN=BI.dim^0.5*(1-1/(4*BI.dim)+1/(21*BI.dim^2));
CMA.pc = zeros(1,BI.dim);
CMA.ps = zeros(1,BI.dim);
CMA.B = eye(BI.dim);
CMA.D = ones(1,BI.dim);
CMA.C = CMA.B * diag(CMA.D.^2) * CMA.B';
CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
CMA.xmean = (BI.xrange(2,:)-BI.xrange(1,:)).*rand(1,BI.dim)+BI.xrange(1,:);
CMA.cy = sqrt(BI.dim)+2*BI.dim/(BI.dim+2);
CMA.delta_sigma_max = 1;

function [CMA,updateInfo] = updateCMAES(CMA,POP,BI)
	[~, Xindex] = sort(-[POP.fit]);
xold = CMA.xmean;
X = cell2mat(arrayfun(@(p)[p.UX p.LX],POP(Xindex(1:CMA.mu))','UniformOutput',false));
	% This is the project's existing length control.  It also provides the
	% required bounded update step for any externally injected parent: its
	% evaluated position remains X, while only its adaptive-update step is
	% Mahalanobis-length clipped.
YRaw = bsxfun(@minus,X,xold)/CMA.sigma;
whitenedNorms = sqrt(sum((YRaw*CMA.invsqrtC').^2,2));
stepScales = min(1,CMA.cy./whitenedNorms);
Y = bsxfun(@times,YRaw,stepScales);
	if nargout > 1
		% These are the exact evaluated parent coordinates, indices and state
		% used below. Returning them is audit-only and never changes CMA-ES.
		updateInfo.eliteIndices = Xindex(1:CMA.mu);
		updateInfo.eliteWeights = CMA.weights;
		updateInfo.parentJoint = X;
		updateInfo.xmeanBefore = xold;
		updateInfo.sigmaBefore = CMA.sigma;
		updateInfo.parentRawSteps = YRaw;
		updateInfo.parentClippedSteps = Y;
		updateInfo.parentWhitenedNorms = whitenedNorms;
		updateInfo.parentStepScales = stepScales;
		updateInfo.stepLengthLimit = CMA.cy;
	end
delta_xmean = CMA.weights * Y;
CMA.xmean = CMA.xmean + delta_xmean * CMA.sigma;
C_mu = Y' * diag(CMA.weights) * Y;
CMA.ps = (1-CMA.cs)*CMA.ps + sqrt(CMA.cs*(2-CMA.cs)*CMA.mueff) * delta_xmean * CMA.invsqrtC;
CMA.pc = (1-CMA.cc)*CMA.pc + sqrt(CMA.cc*(2-CMA.cc)*CMA.mueff) * delta_xmean;
CMA.C = (1-CMA.c1-CMA.cmu) * CMA.C + CMA.c1 * (CMA.pc'*CMA.pc) + CMA.cmu * C_mu; 
delta_sigma = (CMA.cs/CMA.damps)*(norm(CMA.ps)/CMA.chiN - 1);
CMA.sigma = CMA.sigma * exp(min(delta_sigma,CMA.delta_sigma_max));
CMA.C = triu(CMA.C) + triu(CMA.C,1)';
[CMA.B,CMA.D] = eig(CMA.C);
CMA.D = sqrt(diag(CMA.D))';
CMA = repairCMA(CMA);
CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';

function [F,C] = LL_evaluate(xu,xl,fn)
[F,~,C] = llTestProblem(xl,fn,xu);
C = sum(max(0,C));

function [F,C] = UL_evaluate(UPop,LPOP,fn)
[F,~,C] = ulTestProblem(UPop, LPOP, fn);
C = sum(max(0,C));

function reason = getTerminationReason(reachTargetPopulation,reachTargetElite, ...
    reachMaxFEs,reachFlat)
%GETTERMINATIONREASON Keep the original stopping decision but expose why.
if reachTargetPopulation
    reason = 'population_target';
elseif reachTargetElite
    reason = 'elite_target';
elseif reachMaxFEs
    reason = 'maximum_UFE';
elseif reachFlat
    reason = 'flat_progress';
else
    reason = 'continue';
end

function Model = repairCMA(Model)
dim = length(Model.D);
% limit condition of C to 1e14
if any(Model.D<=0)
    Model.D(Model.D<0) = 0;
    tmp = max(Model.D)/1e7;
    Model.C = Model.C + tmp * eye(dim);
    Model.D = Model.D + tmp * ones(1,dim);
end
if max(Model.D) > 1e7 * min(Model.D)
    tmp = max(Model.D)/1e7 - min(Model.D);
    Model.C = Model.C + tmp * eye(dim);
    Model.D = Model.D + tmp * ones(1,dim);
end
% rescale sigma
if Model.sigma > 1e7 * max(Model.D)
    fac = Model.sigma / max(Model.D);
    Model.sigma = Model.sigma / fac;
    Model.D = Model.D * fac;
    Model.pc = Model.pc * fac;
    Model.C = Model.C * fac^2;
end

function diagnostic = initializeMeanTrajectoryDiagnostic(BI)
diagnostic.enabled = false;
diagnostic.rows = zeros(0,3);
diagnostic.columns = {'generation','m1','m2'};

if ~isfield(BI,'meanTrajectoryDiagnostic')
    return;
end

diagnostic = BI.meanTrajectoryDiagnostic;
if ~isfield(diagnostic,'enabled') || ~diagnostic.enabled
    diagnostic.enabled = false;
    return;
end
if BI.u_dim ~= 2
    error('Mean-trajectory diagnostics require exactly two upper variables.');
end

diagnostic.enabled = true;
diagnostic.rows = zeros(0,3);
diagnostic.columns = {'generation','m1','m2'};

function diagnostic = appendMeanTrajectoryDiagnostic(diagnostic, generation, CMA)
% Called only after a completed upper-level mean update (plus generation 0).
% It only reads the first two coordinates of CMA.xmean.
diagnostic.rows(end+1,:) = [generation, CMA.xmean(1:2)];

function diagnostic = initializeGenerationDiagnostic(BI)
% Pure logging configuration. It is inactive by default, and its helpers only
% read variables already produced by the original BL-CMA-ES implementation.
diagnostic.enabled = false;
diagnostic.rows = zeros(0,0);
diagnostic.columns = {};

if ~isfield(BI,'generationDiagnostic')
    return;
end

settings = BI.generationDiagnostic;
if ~isfield(settings,'enabled') || ~settings.enabled
    return;
end

diagnostic.enabled = true;
diagnostic.uDim = BI.u_dim;
diagnostic.lDim = BI.l_dim;
nextColumn = 1;
diagnostic.generationColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.UFEColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.LFETotalColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.LFEGenerationColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.meanColumns = nextColumn:(nextColumn + BI.u_dim - 1);
nextColumn = nextColumn + BI.u_dim;
diagnostic.sigmaColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.generationBestFColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.bestSoFarFColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.generationBestXuColumns = nextColumn:(nextColumn + BI.u_dim - 1);
nextColumn = nextColumn + BI.u_dim;
diagnostic.bestSoFarXuColumns = nextColumn:(nextColumn + BI.u_dim - 1);
nextColumn = nextColumn + BI.u_dim;
diagnostic.populationFMinColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.populationFMedianColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.populationFStdColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.eliteWeightedFColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.feasibleRatioColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.populationCVMinColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.populationCVMedianColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.generationBestCVColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.pcNormColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.psNormColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.condCColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.eigMinCColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.eigMaxCColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.generationBestLLFColumn = nextColumn; nextColumn = nextColumn + 1;
diagnostic.generationBestXlColumns = nextColumn:(nextColumn + BI.l_dim - 1);
nextColumn = nextColumn + BI.l_dim;
diagnostic.meanUpdateAppliedColumn = nextColumn;

meanNames = arrayfun(@(idx)sprintf('m%d',idx),1:BI.u_dim, ...
                     'UniformOutput',false);
generationBestXuNames = arrayfun(@(idx)sprintf('generation_best_xu%d',idx), ...
                                 1:BI.u_dim,'UniformOutput',false);
bestSoFarXuNames = arrayfun(@(idx)sprintf('best_so_far_xu%d',idx), ...
                            1:BI.u_dim,'UniformOutput',false);
generationBestXlNames = arrayfun(@(idx)sprintf('generation_best_xl%d',idx), ...
                                 1:BI.l_dim,'UniformOutput',false);
diagnostic.columns = [ ...
    {'generation','UFE','LFE_total','LFE_generation'}, meanNames, ...
    {'sigma','generation_best_F','best_so_far_F'}, ...
    generationBestXuNames, bestSoFarXuNames, ...
    {'population_F_min','population_F_median','population_F_std', ...
     'elite_weighted_F','feasible_ratio','population_CV_min', ...
     'population_CV_median','generation_best_CV','pc_norm','ps_norm', ...
     'cond_C','eig_min_C','eig_max_C','generation_best_LL_F'}, ...
    generationBestXlNames, {'mean_update_applied'}];
diagnostic.rows = NaN(0,length(diagnostic.columns));
diagnostic.populationGenerationColumn = 1;
diagnostic.populationIndividualIndexColumn = 2;
diagnostic.populationXuColumns = 3:(BI.u_dim + 2);
diagnostic.populationUFColumn = BI.u_dim + 3;
populationXuNames = arrayfun(@(idx)sprintf('xu%d',idx),1:BI.u_dim, ...
                             'UniformOutput',false);
diagnostic.populationColumns = [ ...
    {'generation','individual_index'}, populationXuNames, {'UF'}];
diagnostic.populationRows = NaN(0,length(diagnostic.populationColumns));
diagnostic.covariances = cell(0,1);
diagnostic.updateEliteIndices = cell(0,1);
diagnostic.updateEliteWeights = cell(0,1);
diagnostic.generationBestIndices = zeros(0,1);
% Unlike the optimizer's mutable elite, this snapshot is only used to report
% the literal historical best according to the same original comparator.
diagnostic.bestSoFar = [];

function diagnostic = appendInitialGenerationDiagnostic(diagnostic, CMA, UFE, LFE_total)
% Generation 0 has a defined initial search state, but it has no evaluated
% population, no elite, and hence no current-generation quality metrics.
row = NaN(1,length(diagnostic.columns));
row(diagnostic.generationColumn) = 0;
row(diagnostic.UFEColumn) = UFE;
row(diagnostic.LFETotalColumn) = LFE_total;
row(diagnostic.meanColumns) = CMA.xmean(1:diagnostic.uDim);
row(diagnostic.sigmaColumn) = CMA.sigma;
row = assignCMAStateSummary(row,diagnostic,CMA);
diagnostic = appendGenerationDiagnosticState( ...
    diagnostic,row,CMA,[],[],NaN);

function diagnostic = appendGenerationDiagnostic(diagnostic, generation, CMA, POP, generationBest, generationBestIndex, elite, BI, UFE, LFE_total, LFEAtGenerationStart, meanUpdateApplied, updateInfo)
% generationBest/best index come from the original RF-aware selection.  The
% update elite information is captured from the original updateCMAES sorting.
populationF = [POP.UF];
populationCV = getPopulationConstraintViolations(POP,BI);

% The optimizer's elite can be re-refined in later generations.  Keep a
% separate read-only snapshot so best_so_far means the literal historical best
% under the original constraint-aware upperLevelComparator.  Both original
% sources of an evaluated elite candidate are considered: the RF-aware current
% best and the possibly re-refined optimizer elite.
diagnostic = updateDiagnosticBestSoFar(diagnostic,generationBest,BI);
diagnostic = updateDiagnosticBestSoFar(diagnostic,elite,BI);
bestSoFar = diagnostic.bestSoFar;

if meanUpdateApplied
    eliteIndices = updateInfo.eliteIndices;
    eliteWeights = updateInfo.eliteWeights;
    eliteWeightedF = sum(eliteWeights .* [POP(eliteIndices).UF]);
else
    % The original algorithm does not call updateCMAES on its terminating
    % generation, so no elite group actually pushes a mean update there.
    eliteIndices = [];
    eliteWeights = [];
    eliteWeightedF = NaN;
end
row = NaN(1,length(diagnostic.columns));
row(diagnostic.generationColumn) = generation;
row(diagnostic.UFEColumn) = UFE;
row(diagnostic.LFETotalColumn) = LFE_total;
row(diagnostic.LFEGenerationColumn) = LFE_total - LFEAtGenerationStart;
row(diagnostic.meanColumns) = CMA.xmean(1:diagnostic.uDim);
row(diagnostic.sigmaColumn) = CMA.sigma;
row(diagnostic.generationBestFColumn) = generationBest.UF;
row(diagnostic.bestSoFarFColumn) = bestSoFar.UF;
row(diagnostic.generationBestXuColumns) = generationBest.UX(1:diagnostic.uDim);
row(diagnostic.bestSoFarXuColumns) = bestSoFar.UX(1:diagnostic.uDim);
row(diagnostic.populationFMinColumn) = min(populationF);
row(diagnostic.populationFMedianColumn) = median(populationF);
row(diagnostic.populationFStdColumn) = std(populationF,0);
row(diagnostic.eliteWeightedFColumn) = eliteWeightedF;
row(diagnostic.feasibleRatioColumn) = mean(populationCV == 0);
row(diagnostic.populationCVMinColumn) = min(populationCV);
row(diagnostic.populationCVMedianColumn) = median(populationCV);
row(diagnostic.generationBestCVColumn) = populationCV(generationBestIndex);
row(diagnostic.generationBestLLFColumn) = generationBest.LF;
row(diagnostic.generationBestXlColumns) = generationBest.LX(1:diagnostic.lDim);
row(diagnostic.meanUpdateAppliedColumn) = meanUpdateApplied;
row = assignCMAStateSummary(row,diagnostic,CMA);
diagnostic = appendGenerationDiagnosticState( ...
    diagnostic,row,CMA,eliteIndices,eliteWeights,generationBestIndex);

function diagnostic = appendGenerationPopulationDiagnostic(diagnostic,generation,POP)
% POP already contains the upper vector and its real upper objective value
% after the original lower-level search (and any original refinement).  This
% helper only copies those fields into one row per population member.
populationSize = length(POP);
rows = NaN(populationSize,length(diagnostic.populationColumns));
rows(:,diagnostic.populationGenerationColumn) = generation;
rows(:,diagnostic.populationIndividualIndexColumn) = (1:populationSize)';
for i = 1:populationSize
    rows(i,diagnostic.populationXuColumns) = POP(i).UX(1:diagnostic.uDim);
    rows(i,diagnostic.populationUFColumn) = POP(i).UF;
end
diagnostic.populationRows = [diagnostic.populationRows; rows];

function populationCV = getPopulationConstraintViolations(POP,BI)
% This is exactly the existing CV expression in assignUpperFitness, read from
% already evaluated UC/LC fields.  It does not call an objective or optimizer.
populationCV = [POP.UC];
if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
    populationCV = populationCV + [POP.LC];
end

function diagnostic = updateDiagnosticBestSoFar(diagnostic,candidate,BI)
if isempty(candidate)
    return;
end
if isempty(diagnostic.bestSoFar) || ...
        upperLevelComparator(candidate,diagnostic.bestSoFar,BI)
    diagnostic.bestSoFar = candidate;
end

function row = assignCMAStateSummary(row,diagnostic,CMA)
% cond/eig are numerical summaries of the existing C only; they do not affect
% the optimizer state, its covariance update, or any function-evaluation count.
eigenvalues = eig(CMA.C);
row(diagnostic.pcNormColumn) = norm(CMA.pc);
row(diagnostic.psNormColumn) = norm(CMA.ps);
row(diagnostic.condCColumn) = cond(CMA.C);
row(diagnostic.eigMinCColumn) = min(real(eigenvalues));
row(diagnostic.eigMaxCColumn) = max(real(eigenvalues));

function diagnostic = appendGenerationDiagnosticState(diagnostic,row,CMA,updateEliteIndices,updateEliteWeights,generationBestIndex)
diagnostic.rows(end+1,:) = row;
diagnostic.covariances{end+1,1} = CMA.C;
diagnostic.updateEliteIndices{end+1,1} = updateEliteIndices;
diagnostic.updateEliteWeights{end+1,1} = updateEliteWeights;
diagnostic.generationBestIndices(end+1,1) = generationBestIndex;
