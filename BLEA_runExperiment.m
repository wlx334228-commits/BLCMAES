% run BLEAQ2
% cd D:\soft\BLEAQ2\bleaq2-master 
% cd D:\soft\NBLEA
% cd D:\soft\BLEAQ
% cd 'D:\workspace\Computational Intelligence'

pwd

maxRuns = 20;
BI_list = [];
BI_list = [BI_list;getBLOPinfo('SMD',1:12,20)];
% BI_list = [BI_list;getBLOPinfo('TP',1:10)];
% BI_list = [BI_list;getBLOPinfo('SMD',1:12,10)];
% BI_list = [BI_list;getBLOPinfo('SMD',1:12,20)];

% BI_list = [BI_list;getBLOPinfo('GoldMining')];
% BI_list = [BI_list;getBLOPinfo('DecisionMaking')];

% BI_list = [BI_list;getBLOPinfo('TP',1:10)];
% BI_list = [BI_list;getBLOPinfo('SMD',9:12,5)];
% BI_list = [BI_list;getBLOPinfo('TP',10)];

algName = 'BLCMAES';
% Results are retained for every independent run.  The saved summary uses
% the same median (IQR) convention as Tables S-I and S-II in supplement.pdf.
results = struct();
for BI = BI_list'
    if BI.dim == 3
        % for gold mining problem
        BI.UmaxFEs = 1500;
        BI.UmaxImprFEs = 70;
        BI.LmaxFEs = 150;
        BI.LmaxImprFEs = 15;
    elseif BI.dim == 5 || BI.dim == 6 || strcmp(BI.fn(1:2),'tp')
        % for generic benchmark test problem or decision making problem 
        BI.UmaxFEs = 2500;
        BI.UmaxImprFEs = 350;
        BI.LmaxFEs = 250;
        BI.LmaxImprFEs = 25;
    elseif BI.dim == 10
        BI.UmaxFEs = 3500;
        BI.UmaxImprFEs = 500;
        BI.LmaxFEs = 350;
        BI.LmaxImprFEs = 35;
    elseif BI.dim == 20
        BI.UmaxFEs = 5000;
        BI.UmaxImprFEs = 750;
        BI.LmaxFEs = 500;
        BI.LmaxImprFEs = 50;
    else
        error('unknown dimensionality');
    end
    
    BI.u_N = 50;
    BI.l_N = 50;

% 	parfor runNo = 1:maxRuns
	% Columns: UF, LF, UAcc, LAcc, UFEs, LFEs, runTimeSeconds.
	runResults = zeros(maxRuns,7);
    for runNo = 1:maxRuns
		tic;
% 		ins = wapper(BI,true);
        ins = BLCMAES(BI);
		ins.runPath = pwd;
		ins.runNo = runNo;
		ins.BI = BI;
		ins.alg = algName;
		ins.runTime = toc;
		% The paper reports accuracies no smaller than the tolerance (1e-6).
		uAcc = max(abs(ins.UF - BI.u_fopt), BI.u_ftol);
		lAcc = max(abs(ins.LF - BI.l_fopt), BI.l_ftol);
		runResults(runNo,:) = [ins.UF, ins.LF, uAcc, lAcc, ...
		                       ins.UFEs, ins.LFEs, ins.runTime];
	        fprintf(['%s %s #%d: UF=%g (err=%g), LF=%g (err=%g), ' ...
	                 'UFEs=%d, LFEs=%d, time=%.2fs\n'], ...
	                ins.alg, ins.BI.fn, ins.runNo, ...
	                ins.UF, abs(ins.UF - BI.u_fopt), ...
	                ins.LF, abs(ins.LF - BI.l_fopt), ...
	                ins.UFEs, ins.LFEs, ins.runTime);
	end	

	resultColumns = {'UF','LF','UAcc','LAcc','UFEs','LFEs','runTimeSeconds'};
	resultMedian = median(runResults,1);
	resultIQR = quantile(runResults,0.75,1) - quantile(runResults,0.25,1);
	results.(BI.fn) = struct('BI',BI, 'columns',{resultColumns}, ...
	                         'runs',runResults, 'median',resultMedian, ...
	                         'iqr',resultIQR);

	fprintf(['%s summary, median (IQR): UAcc=%0.3E (%0.3E), ' ...
	         'LAcc=%0.3E (%0.3E), UFEs=%0.3E (%0.3E), ' ...
	         'LFEs=%0.3E (%0.3E)\n'], ...
	        BI.fn, resultMedian(3), resultIQR(3), ...
	        resultMedian(4), resultIQR(4), ...
	        resultMedian(5), resultIQR(5), ...
	        resultMedian(6), resultIQR(6));

	% Save after every problem so completed experiments survive interruption.
	% BI.dim is the total dimension: 5 means (2+3), 10 means (5+5),
	% and 20 means (10+10).
	resultFile = sprintf('BLCMAES_SMD_%dD_results.mat', BI.dim);
	save(resultFile,'results','maxRuns');
end



