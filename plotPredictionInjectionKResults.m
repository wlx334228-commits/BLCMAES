function plotPredictionInjectionKResults(outputRoot)
%PLOTPREDICTIONINJECTIONKRESULTS Draw the two requested K-ablation figures.
% Reads only the completed summary CSV.  It never reruns an optimisation or
% evaluates a function.

if nargin < 1 || isempty(outputRoot)
    error('BLCMAES:PredictionInjectionPlot', ...
        'Specify a completed K-ablation result directory.');
end
summaryFile = fullfile(outputRoot,'injection_summary.csv');
assert(exist(summaryFile,'file') == 2, ...
    'Missing summary file: %s',summaryFile);
summary = readtable(summaryFile);
required = {'SMD','prediction_inject_count','UFE_median','UFE_IQR', ...
    'LFE_median','LFE_IQR','upper_error_median','lower_error_median', ...
    'upper_success_rate'};
assert(all(ismember(required,summary.Properties.VariableNames)), ...
    'The summary CSV does not contain K-ablation fields.');

figureDir = fullfile(outputRoot,'figures');
if ~exist(figureDir,'dir'), mkdir(figureDir); end
for smdNo = unique(summary.SMD)'
    rows = summary(summary.SMD == smdNo,:);
    rows = sortrows(rows,'prediction_inject_count');
    K = rows.prediction_inject_count;

    figureHandle = figure('Visible','off','Color','w','Position',[100 100 820 640]);
    subplot(2,1,1);
    errorbar(K,rows.UFE_median,rows.UFE_IQR/2,'-o', ...
        'LineWidth',1.3,'MarkerSize',5);
    grid on;
    xlabel('Prediction injection count K');
    ylabel('UFE (median \pm IQR/2)');
    title(sprintf('SMD%d: K versus upper-level evaluations',smdNo));
    xlim([min(K)-0.25,max(K)+0.25]);
    xticks(K);
    subplot(2,1,2);
    errorbar(K,rows.LFE_median,rows.LFE_IQR/2,'-s', ...
        'LineWidth',1.3,'MarkerSize',5);
    grid on;
    xlabel('Prediction injection count K');
    ylabel('LFE (median \pm IQR/2)');
    title(sprintf('SMD%d: K versus lower-level evaluations',smdNo));
    xlim([min(K)-0.25,max(K)+0.25]);
    xticks(K);
    saveas(figureHandle,fullfile(figureDir, ...
        sprintf('SMD%d_K_evaluations.png',smdNo)));
    close(figureHandle);

    figureHandle = figure('Visible','off','Color','w','Position',[100 100 820 760]);
    subplot(3,1,1);
    semilogy(K,max(rows.upper_error_median,realmin),'-o', ...
        'LineWidth',1.3,'MarkerSize',5);
    grid on;
    xlabel('Prediction injection count K');
    ylabel('Upper error (median)');
    title(sprintf('SMD%d: K versus final upper-level accuracy',smdNo));
    xlim([min(K)-0.25,max(K)+0.25]);
    xticks(K);
    subplot(3,1,2);
    semilogy(K,max(rows.lower_error_median,realmin),'-s', ...
        'LineWidth',1.3,'MarkerSize',5);
    grid on;
    xlabel('Prediction injection count K');
    ylabel('Lower error (median)');
    title(sprintf('SMD%d: K versus final lower-level accuracy',smdNo));
    xlim([min(K)-0.25,max(K)+0.25]);
    xticks(K);
    subplot(3,1,3);
    plot(K,rows.upper_success_rate,'-d','LineWidth',1.3,'MarkerSize',5);
    grid on;
    ylim([0,1.05]);
    xlabel('Prediction injection count K');
    ylabel('Upper success rate');
    title(sprintf('SMD%d: K versus upper-level success rate',smdNo));
    xlim([min(K)-0.25,max(K)+0.25]);
    xticks(K);
    saveas(figureHandle,fullfile(figureDir, ...
        sprintf('SMD%d_K_accuracy_success.png',smdNo)));
    close(figureHandle);
end
end
