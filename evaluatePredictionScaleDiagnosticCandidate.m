function result = evaluatePredictionScaleDiagnosticCandidate(xu,state,BI,stream)
%EVALUATEPREDICTIONSCALEDIAGNOSTICCANDIDATE Isolated true BL evaluation.
% The lower-level optimizer below is a stream-isolated transcription of the
% existing BLCMAES lowerLevelSearch.  Its only intentional difference is
% that every randn call is made on the supplied private RandStream, so this
% diagnostic cannot alter the main CMA-ES random trajectory.

if nargin < 4 || isempty(stream)
    stream = RandStream('mt19937ar','Seed',1);
end
global ulFunctionEvaluations llFunctionEvaluations;
UFEbefore = ulFunctionEvaluations;
LFEbefore = llFunctionEvaluations;
[xl,LF,LC,RF] = isolatedLowerLevelSearch(xu,state,BI,stream);
[UF,UC] = isolatedUpperEvaluate(xu,xl,BI.fn);
result = struct('xu',xu,'xl',xl,'UF',UF,'UC',UC,'LF',LF,'LC',LC, ...
    'RF',RF,'diagnostic_UFE',ulFunctionEvaluations-UFEbefore, ...
    'diagnostic_LFE',llFunctionEvaluations-LFEbefore);
end

function [bestLX,bestLF,bestLC,bestRF] = isolatedLowerLevelSearch(xu,state,BI,stream)
sigma0 = 1;
CMA = struct('xmean',state.cma_mean,'sigma',state.sigma, ...
    'C',state.C,'pc',state.pc,'delta_sigma_max',1);
LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
LCMA.sigma = sigma0;
LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end)*CMA.sigma^2;
LCMA.pc = CMA.pc(BI.u_dim+1:end)*CMA.sigma;
LCMA.ps = zeros(1,BI.l_dim);
lambda = 4+floor(3*log(BI.l_dim));
mu = floor(lambda/2);
weights = log(mu+1/2)-log(1:mu); weights = weights/sum(weights);
mueff = sum(weights)^2/sum(weights.^2);
cc = (4+mueff/BI.l_dim)/(BI.l_dim+4+2*mueff/BI.l_dim);
cs = (mueff+2)/(BI.l_dim+mueff+5);
c1 = 2/((BI.l_dim+1.3)^2+mueff);
cmu = min(1-c1,2*(mueff-2+1/mueff)/((BI.l_dim+2)^2+mueff));
damps = 1+2*max(0,sqrt((mueff-1)/(BI.l_dim+1))-1)+cs;
chiN = BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
[LCMA.B,LCMA.D] = eig(LCMA.C); LCMA.D = sqrt(diag(LCMA.D))';
LCMA.invsqrtC = LCMA.B*diag(LCMA.D.^-1)*LCMA.B';
cy = sqrt(BI.l_dim)+2*BI.l_dim/(BI.l_dim+2);
bestIndv = []; bestRF = false;
maxIter = ceil(BI.LmaxFEs/lambda);
ImprIter = ceil(BI.LmaxImprFEs/lambda);
record = zeros(1,maxIter);
for iter = 1:maxIter
    for index = 1:lambda
        Q(index).LX = LCMA.xmean+LCMA.sigma*randn(stream,1,BI.l_dim) ...
            .*LCMA.D*LCMA.B';
        high = Q(index).LX > BI.l_ub;
        Q(index).LX(high) = (LCMA.xmean(high)+BI.l_ub(high))/2;
        low = Q(index).LX < BI.l_lb;
        Q(index).LX(low) = (LCMA.xmean(low)+BI.l_lb(low))/2;
        [Q(index).LF,Q(index).LC] = isolatedLowerEvaluate(xu,Q(index).LX,BI.fn);
    end
    Q = assignIsolatedLowerFitness(Q);
    [~,order] = sort(-[Q.fit]);
    xold = LCMA.xmean;
    X = cell2mat(arrayfun(@(item)item.LX,Q','UniformOutput',false));
    Y = bsxfun(@minus,X(order(1:mu),:),xold)/LCMA.sigma;
    Y = bsxfun(@times,Y,min(1,cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
    deltaMean = weights*Y;
    LCMA.xmean = LCMA.xmean+deltaMean*LCMA.sigma;
    Cmu = Y'*diag(weights)*Y;
    LCMA.ps = (1-cs)*LCMA.ps+sqrt(cs*(2-cs)*mueff)*deltaMean*LCMA.invsqrtC;
    LCMA.pc = (1-cc)*LCMA.pc+sqrt(cc*(2-cc)*mueff)*deltaMean;
    LCMA.C = (1-c1-cmu)*LCMA.C+c1*(LCMA.pc'*LCMA.pc)+cmu*Cmu;
    deltaSigma = (cs/damps)*(norm(LCMA.ps)/chiN-1);
    LCMA.sigma = LCMA.sigma*exp(min(CMA.delta_sigma_max,deltaSigma));
    LCMA.C = triu(LCMA.C)+triu(LCMA.C,1)';
    [LCMA.B,LCMA.D] = eig(LCMA.C); LCMA.D = sqrt(diag(LCMA.D))';
    LCMA = repairIsolatedCMA(LCMA);
    LCMA.invsqrtC = LCMA.B*diag(LCMA.D.^-1)*LCMA.B';
    if isBetterLower(Q(order(1)),bestIndv)
        bestIndv = Q(order(1));
    end
    record(iter) = bestIndv.LF;
    if (iter > ImprIter && abs(record(iter)-record(iter-ImprIter+1))/ ...
            (abs(record(1))+abs(record(iter))) < 1e-4) || ...
            (iter > ImprIter && abs(record(iter)-record(iter-ImprIter+1)) < ...
            10*BI.l_ftol) || LCMA.sigma/sigma0 < 1e-2 || ...
            LCMA.sigma/sigma0 > 1e2
        bestRF = true;
        break;
    end
end
bestLX = bestIndv.LX; bestLF = bestIndv.LF; bestLC = bestIndv.LC;
end

function [F,C] = isolatedLowerEvaluate(xu,xl,fn)
[F,~,C] = llTestProblem(xl,fn,xu); C = sum(max(0,C));
end

function [F,C] = isolatedUpperEvaluate(xu,xl,fn)
[F,~,C] = ulTestProblem(xu,xl,fn); C = sum(max(0,C));
end

function Q = assignIsolatedLowerFitness(Q)
fitness = combineFitness([Q.LF],[Q.LC]);
for index = 1:numel(Q), Q(index).fit = fitness(index); end
end

function better = isBetterLower(P,Q)
if isempty(Q), better = true; return; end
fitness = combineFitness([P.LF Q.LF],[P.LC Q.LC]);
better = fitness(1) >= fitness(2);
end

function fit = combineFitness(objective,constraint)
if all(constraint == 0)
    fit = objective;
else
    fit = -constraint;
    feasible = constraint == 0;
    if any(feasible)
        fit(feasible) = objective(feasible);
        fit(~feasible) = min(fit(feasible))-constraint(~feasible);
    end
end
end

function model = repairIsolatedCMA(model)
dimension = length(model.D);
if any(model.D <= 0)
    model.D(model.D < 0) = 0;
    increment = max(model.D)/1e7;
    model.C = model.C+increment*eye(dimension);
    model.D = model.D+increment*ones(1,dimension);
end
if max(model.D) > 1e7*min(model.D)
    increment = max(model.D)/1e7-min(model.D);
    model.C = model.C+increment*eye(dimension);
    model.D = model.D+increment*ones(1,dimension);
end
if model.sigma > 1e7*max(model.D)
    factor = model.sigma/max(model.D);
    model.sigma = model.sigma/factor;
    model.D = model.D*factor;
    model.pc = model.pc*factor;
    model.C = model.C*factor^2;
end
end
