function [models,shared] = fitPredictionQualityModels(event,BI,options,seed,nextNormal)
%FITPREDICTIONQUALITYMODELS One archive selection and search pool for all models.
% QP/CP/RBF/GPR maximise UF, using only already evaluated history.
% QP_analytic retains the original negative-Hessian stationary-point reference.
ids={'QP','CP','RBF','GPR','QP_analytic'};
savedRng=rng; rngCleanup=onCleanup(@() rng(savedRng)); %#ok<NASGU>
rng(seed,'twister');
models=repmat(emptyModel(),5,1);
for i=1:5, models(i).id=ids{i}; end
meanUpper=event.CMA.xmean(1:BI.u_dim);
opts=predictionInjectionOptions(BI);
legacyClock=tic;
legacy=fitLocalUpperQuadratic(event.history,meanUpper,event.generation,opts);
legacySeconds=toc(legacyClock);
shared=struct('selectedSamples',legacy.selectedSamples,'scale',legacy.scale, ...
    'mean',meanUpper,'z',zeros(0,BI.u_dim),'y',zeros(0,1), ...
    'yMean',NaN,'yScale',NaN,'zLower',NaN(1,BI.u_dim), ...
    'zUpper',NaN(1,BI.u_dim),'radius',options.localRadius, ...
    'candidateZ',zeros(0,BI.u_dim),'candidateX',zeros(0,BI.u_dim), ...
    'poolSeed',seed,'selectionReason',legacy.reason,'selectionAndQPFitSeconds',legacySeconds);
if size(legacy.selectedSamples,1)~=options.sampleCount || ...
        ~isfinite(legacy.scale) || legacy.scale<= ...
        opts.minScaleRelative*max(1,norm(meanUpper))
    for i=1:5, models(i).reason=legacy.reason; end
    return;
end
shared.z=bsxfun(@minus,legacy.selectedSamples(:,3:4),meanUpper)/legacy.scale;
rawY=legacy.selectedSamples(:,5);
shared.yMean=mean(rawY); shared.yScale=std(rawY);
if ~isfinite(shared.yScale) || shared.yScale<=eps(max(1,max(abs(rawY))))
    for i=1:5, models(i).reason='degenerate_response_scale'; end
    return;
end
shared.y=(rawY-shared.yMean)/shared.yScale;
shared.zLower=(BI.u_lb-meanUpper)/legacy.scale;
shared.zUpper=(BI.u_ub-meanUpper)/legacy.scale;
shared.candidateZ=makeCommonPool(shared,options);
shared.candidateX=meanUpper+legacy.scale*shared.candidateZ;
for i=1:4
    m=models(i); clock=tic;
    try
        m=fitOne(m,shared,legacy,options);
    catch exception
        m.reason=['fit_failed:' exception.identifier]; m.failureMessage=exception.message;
    end
    m.fitTime=toc(clock);
    if i==1, m.fitTime=m.fitTime+legacySeconds; end
    if m.fitValid
        clock=tic;
        try
            m=optimiseOne(m,shared,options);
        catch exception
            m.reason=['optimization_failed:' exception.identifier];
            m.failureMessage=exception.message;
        end
        m.optimizationTime=toc(clock);
    end
    if m.valid
        m=finishPrediction(m,shared,BI,options);
    end
    models(i)=m;
end
% Keep the legacy reference's stationary-point rule and existing upper repair.
m=models(5); m.fitTime=0; % Already fitted once for QP; no duplicate fit cost.
m.fitValid=all(isfinite(legacy.coefficients));
m.reason=legacy.reason; m.H=legacy.H;
m.payload=struct('kind','QP','coefficients',legacy.coefficients);
if legacy.valid
    x=legacy.x_pred; m.requestedX=x;
    reference=nextNormal(end).UX;
    high=x>BI.u_ub; x(high)=(reference(high)+BI.u_ub(high))/2;
    low=x<BI.u_lb; x(low)=(reference(low)+BI.u_lb(low))/2;
    m.boundaryRepaired=any(high|low);
    m.x=x; m.z=(x-meanUpper)/legacy.scale;
    m.predictedUF=polyDesign(m.z,2)*legacy.coefficients;
    m.valid=all(isfinite(x))&&all(x>=BI.u_lb)&&all(x<=BI.u_ub);
    m.insideCommonRegion=norm(m.z)<=options.localRadius+1e-8;
    m.duplicateHistory=any(vecnorm(shared.z-m.z,2,2)<=options.duplicateTolerance);
    if m.valid, m.reason='valid'; else, m.reason='invalid_analytic_prediction'; end
end
models(5)=m;
for i=1:5
    if models(i).valid
        for j=1:i-1
            if models(j).valid && norm(models(i).z-models(j).z)<=options.duplicateTolerance
                models(i).duplicateEarlierModel=true;
            end
        end
    end
end
end

function m=emptyModel()
m=struct('id','','fitValid',false,'valid',false,'reason','not_fitted', ...
    'failureMessage','','x',[NaN NaN],'requestedX',[NaN NaN],'z',[NaN NaN], ...
    'predictedUF',NaN,'predictedStd',NaN,'fitTime',0,'optimizationTime',0, ...
    'fitRMSE',NaN,'designRank',NaN,'designCondition',NaN,'regularization',0, ...
    'H',NaN(2),'payload',struct(),'startIndices',zeros(0,1), ...
    'startZ',zeros(0,2),'exitFlags',zeros(0,1), ...
    'optimizerFunctionCounts',zeros(0,1),'optimizerIterations',zeros(0,1), ...
    'optimizerSolutions',zeros(0,2),'successfulStarts',0,'surrogateEvaluations',0, ...
    'duplicateHistory',false,'duplicateEarlierModel',false, ...
    'duplicateOptimizerSolutions',0,'boundaryRepaired',false, ...
    'atBoundary',false,'insideCommonRegion',false,'gpMeanCheckError',NaN, ...
    'gpWarnings','');
end

function pool=makeCommonPool(shared,options)
stream=RandStream('mt19937ar','Seed',shared.poolSeed);
pool=zeros(options.candidateCount,2); count=0; attempts=0;
while count<options.candidateCount && attempts<2000
    direction=randn(stream,256,2);
    direction=direction./max(vecnorm(direction,2,2),realmin);
    z=options.localRadius*sqrt(rand(stream,256,1)).*direction;
    good=all(z>=shared.zLower & z<=shared.zUpper,2);
    z=z(good,:); take=min(size(z,1),options.candidateCount-count);
    pool(count+(1:take),:)=z(1:take,:); count=count+take; attempts=attempts+1;
end
assert(count==options.candidateCount,'BLCMAES:ModelQualityPool', ...
    'Cannot construct the shared feasible local candidate pool.');
end

function m=fitOne(m,s,legacy,options)
z=s.z; y=s.y;
switch m.id
    case 'QP'
        if any(~isfinite(legacy.coefficients)) || legacy.designRank<6 || ...
                legacy.designCondition>1e10
            m.reason='invalid_quadratic_design'; return;
        end
        coef=legacy.coefficients; coef(1)=coef(1)-s.yMean; coef=coef/s.yScale;
        m.payload=struct('kind','QP','coefficients',coef);
        m.H=legacy.H; m.designRank=legacy.designRank; m.designCondition=legacy.designCondition;
    case 'CP'
        D=polyDesign(z,3); penalty=diag([0 ones(1,9)]);
        lambda=options.ridgeRelative*norm(D,'fro')^2/size(D,2);
        augmented=[D;sqrt(lambda)*penalty];
        coef=augmented\[y;zeros(10,1)];
        m.payload=struct('kind','CP','coefficients',coef);
        m.regularization=lambda; m.designRank=rank(D); m.designCondition=cond(augmented);
    case 'RBF'
        P=[ones(size(z,1),1) z];
        if rank(P)<3, m.reason='rank_deficient_rbf_tail'; return; end
        K=vecnorm(reshape(z,[],1,2)-reshape(z,1,[],2),2,3).^3;
        Z=null(P'); projected=Z'*K*Z;
        lambda=options.ridgeRelative*max(1,norm(projected,2));
        regularized=(projected+projected')/2+lambda*eye(size(projected));
        w=Z*(regularized\(Z'*y));
        beta=P\(y-(K+lambda*eye(size(K)))*w);
        assert(norm(P'*w)<1e-6*max(1,norm(w)),'Invalid RBF tail constraint.');
        m.payload=struct('kind','RBF','centres',z,'weights',w,'beta',beta);
        m.regularization=lambda; m.designRank=rank(P); m.designCondition=cond(regularized);
    case 'GPR'
        lastwarn('');
        gp=fitrgp(z,y,'KernelFunction','matern52','BasisFunction','constant', ...
            'FitMethod','exact','PredictMethod','exact','Standardize',false, ...
            'Sigma',.01,'ConstantSigma',true,'SigmaLowerBound',1e-6,'Optimizer','quasinewton', ...
            'OptimizerOptions',statset('MaxIter',options.gpMaxIterations,'Display','off'));
        [warningText,warningID]=lastwarn;
        m.gpWarnings=[warningID ':' warningText];
        m.payload=struct('kind','GPR','centres',z,'alpha',gp.Alpha, ...
            'beta',gp.Beta,'parameters',gp.KernelInformation.KernelParameters,'gp',gp);
        cached=valueGradient(m.payload,s.candidateZ);
        native=predict(gp,s.candidateZ);
        m.gpMeanCheckError=max(abs(cached-native));
        assert(m.gpMeanCheckError<=1e-7*max(1,max(abs(native))), ...
            'BLCMAES:GPRMeanMismatch','Cached GPR mean differs from fitrgp predict.');
        m.regularization=gp.Sigma^2;
end
fit=valueGradient(m.payload,z);
assert(all(isfinite(fit)),'Non-finite fitted response.');
m.fitRMSE=s.yScale*sqrt(mean((fit-y).^2));
m.fitValid=true; m.reason='fitted';
end

function m=optimiseOne(m,s,options)
values=valueGradient(m.payload,s.candidateZ);
assert(all(isfinite(values)),'Invalid shared-pool predictions.');
[~,order]=sortrows([-values,(1:numel(values))'],[1 2]);
m.startIndices=order(1:options.startCount);
m.startZ=s.candidateZ(m.startIndices,:);
solverOptions=optimoptions('fmincon','Algorithm','sqp','Display','off', ...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
    'MaxIterations',options.maxIterations, ...
    'MaxFunctionEvaluations',options.maxFunctionEvaluations, ...
    'OptimalityTolerance',1e-7,'StepTolerance',1e-9,'ConstraintTolerance',1e-9);
best=-Inf; m.exitFlags=zeros(5,1); m.optimizerSolutions=NaN(5,2);
m.optimizerFunctionCounts=zeros(5,1); m.optimizerIterations=zeros(5,1);
for j=1:options.startCount
    [candidate,negative,flag,output]=fmincon( ...
        @(x) negativeModel(m.payload,x),m.startZ(j,:)',[],[],[],[], ...
        s.zLower',s.zUpper',@(x) ballConstraint(x,options.localRadius),solverOptions);
    candidate=candidate(:)';
    m.exitFlags(j)=flag; m.optimizerFunctionCounts(j)=output.funcCount;
    m.optimizerIterations(j)=output.iterations; m.optimizerSolutions(j,:)=candidate;
    feasible=all(isfinite(candidate))&&all(candidate>=s.zLower-1e-8) && ...
        all(candidate<=s.zUpper+1e-8)&&norm(candidate)<=options.localRadius+1e-8;
    if flag>0 && feasible && isfinite(negative)
        m.successfulStarts=m.successfulStarts+1;
        if -negative>best
            best=-negative; m.z=candidate;
        end
    end
end
m.surrogateEvaluations=options.candidateCount+sum(m.optimizerFunctionCounts);
if m.successfulStarts==0, m.reason='no_converged_valid_multistart'; return; end
m.x=s.mean+s.scale*m.z; m.requestedX=m.x;
m.predictedUF=s.yMean+s.yScale*best;
for j=2:5
    if all(isfinite(m.optimizerSolutions(j,:))) && any(vecnorm( ...
            m.optimizerSolutions(1:j-1,:)-m.optimizerSolutions(j,:),2,2)<options.duplicateTolerance)
        m.duplicateOptimizerSolutions=m.duplicateOptimizerSolutions+1;
    end
end
m.valid=true; m.reason='valid';
if strcmp(m.id,'GPR')
    [native,sd]=predict(m.payload.gp,m.z);
    assert(abs(native-best)<1e-7*max(1,abs(native)),'GPR optimum mean mismatch.');
    m.predictedStd=s.yScale*sd;
end
end

function m=finishPrediction(m,s,BI,options)
tolerance=1e-8*max(1,s.scale);
if any(~isfinite(m.x))||any(m.x<BI.u_lb-tolerance)||any(m.x>BI.u_ub+tolerance) || ...
        norm(m.z)>options.localRadius+1e-8
    m.valid=false; m.reason='invalid_shared_region_prediction'; return;
end
m.duplicateHistory=any(vecnorm(s.z-m.z,2,2)<=options.duplicateTolerance);
m.insideCommonRegion=true;
m.atBoundary=abs(norm(m.z)-options.localRadius)<1e-6 || ...
    any(abs(m.z-s.zLower)<1e-6 | abs(m.z-s.zUpper)<1e-6);
end

function [f,g]=negativeModel(payload,x)
[value,gradient]=valueGradient(payload,x(:)');
f=-value; g=-gradient(:);
end

function [c,ceq,gc,gceq]=ballConstraint(x,radius)
c=sum(x.^2)-radius^2; ceq=[]; gc=2*x(:); gceq=[];
end

function [value,gradient]=valueGradient(payload,z)
switch payload.kind
    case {'QP','CP'}
        degree=2+strcmp(payload.kind,'CP');
        value=polyDesign(z,degree)*payload.coefficients;
        if nargout>1
            x=z(1); y=z(2); a=payload.coefficients;
            gradient=[a(2)+2*a(4)*x+a(5)*y, a(3)+a(5)*x+2*a(6)*y];
            if degree==3
                gradient=gradient+[3*a(7)*x^2+2*a(8)*x*y+a(9)*y^2, ...
                    a(8)*x^2+2*a(9)*x*y+3*a(10)*y^2];
            end
        end
    case 'RBF'
        delta=reshape(z,[],1,2)-reshape(payload.centres,1,[],2);
        distance=vecnorm(delta,2,3);
        value=distance.^3*payload.weights+[ones(size(z,1),1) z]*payload.beta;
        if nargout>1
            gradient=3*sum(payload.weights.*distance'.*(z-payload.centres),1)+payload.beta(2:3)';
        end
    case 'GPR'
        delta=reshape(z,[],1,2)-reshape(payload.centres,1,[],2);
        distance=vecnorm(delta,2,3);
        c=sqrt(5)/payload.parameters(1); variance=payload.parameters(2)^2;
        kernel=variance*(1+c*distance+c^2*distance.^2/3).*exp(-c*distance);
        value=kernel*payload.alpha+payload.beta;
        if nargout>1
            factor=-variance*c^2/3*(1+c*distance').*exp(-c*distance').*payload.alpha;
            gradient=sum(factor.*(z-payload.centres),1);
        end
end
end

function D=polyDesign(z,degree)
x=z(:,1); y=z(:,2);
D=[ones(size(x)),x,y,x.^2,x.*y,y.^2];
if degree==3, D=[D,x.^3,x.^2.*y,x.*y.^2,y.^3]; end
end
