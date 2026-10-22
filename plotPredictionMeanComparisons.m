function outputDir = plotPredictionMeanComparisons(resultRoot)
%PLOTPREDICTIONMEANCOMPARISONS Compare paired 2-D mean trajectories.
% This script is post-processing only: it reads saved MAT results and never
% calls the optimiser, an upper/lower objective, or a random search routine.

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(resultRoot)
    resultRoot = fullfile(projectRoot,'results_prediction_comparison', ...
        '20261021_155629');
end
assert(exist(resultRoot,'dir') == 7,'Result directory does not exist: %s',resultRoot);

baseOutputDir = fullfile(resultRoot,'mean_trajectory_comparison');
outputDir = baseOutputDir;
if exist(outputDir,'dir') == 7
    outputDir = [baseOutputDir '_' datestr(now,'yyyymmdd_HHMMSS')];
end
mkdir(outputDir);

% Fixed seed makes the random two-run selection per SMD reproducible.
previousRng = rng;
cleanupRng = onCleanup(@()rng(previousRng)); %#ok<NASGU>
rng(20261021,'twister');
selectedRuns = zeros(8,2);
for smdNo = 1:8
    selectedRuns(smdNo,:) = sort(randperm(10,2));
end
writetable(table((1:8)',selectedRuns(:,1),selectedRuns(:,2), ...
    'VariableNames',{'SMD','run_a','run_b'}), ...
    fullfile(outputDir,'selected_runs.csv'));

for smdNo = 1:8
    figureHandle = figure('Visible','off','Color','w', ...
        'Position',[100,100,1560,820]);
    layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    title(layout,sprintf(['SMD%d paired CMA-ES mean trajectories: original ', ...
        'vs 0.5/0.5 prediction blend'],smdNo), ...
        'FontWeight','bold');

    for panel = 1:2
        runNo = selectedRuns(smdNo,panel);
        baselineFile = resultFile(resultRoot,smdNo,'baseline',runNo);
        predictionFile = resultFile(resultRoot,smdNo,'prediction',runNo);
        baselineData = load(baselineFile,'ins');
        predictionData = load(predictionFile,'ins');
        baseline = extractMeanPath(baselineData.ins.predictionLog);
        corrected = extractMeanPath(predictionData.ins.predictionLog);
        seed = (smdNo-1)*10 + runNo;

        fullAxis = nexttile(layout,panel);
        plotMeanComparison(fullAxis,baseline,corrected,false);
        title(fullAxis,sprintf('run %02d, seed %d: full trajectory',runNo,seed));
        if panel == 1
            legend(fullAxis,'Location','best','FontSize',8);
        end

        localAxis = nexttile(layout,panel+2);
        plotMeanComparison(localAxis,baseline,corrected,true);
        title(localAxis,sprintf('run %02d, seed %d: generations 9 onward', ...
            runNo,seed));
        if panel == 1
            legend(localAxis,'Location','best','FontSize',8);
        end
    end

    exportgraphics(figureHandle,fullfile(outputDir, ...
        sprintf('SMD%d_mean_trajectory_comparison.png',smdNo)), ...
        'Resolution',220);
    close(figureHandle);
end

fprintf('Saved eight paired mean-trajectory figures in:\n%s\n',outputDir);
end

function file = resultFile(resultRoot,smdNo,variant,runNo)
file = fullfile(resultRoot,sprintf('SMD%d',smdNo),'DU2_DL3',variant, ...
    sprintf('SMD%d_DU2_DL3_run%02d.mat',smdNo,runNo));
assert(exist(file,'file') == 2,'Missing result file: %s',file);
end

function trace = extractMeanPath(predictionLog)
% Each row describes m_before used at generation g and m_new used next.
trace.points = zeros(0,3); % [generation, m1, m2]
trace.events = zeros(0,7); % [g, mCMA1, mCMA2, mNew1, mNew2, xPred1, xPred2]
for row = 1:numel(predictionLog)
    item = predictionLog(row);
    if row == 1 && all(isfinite(item.m_before))
        trace.points(end+1,:) = [0,item.m_before]; %#ok<AGROW>
    end
    if item.mean_update_applied && all(isfinite(item.m_new))
        trace.points(end+1,:) = [item.generation,item.m_new]; %#ok<AGROW>
    end
    if item.correction_applied && all(isfinite(item.m_cma)) && ...
            all(isfinite(item.m_new)) && all(isfinite(item.x_pred))
        trace.events(end+1,:) = [item.generation,item.m_cma,item.m_new, ...
            item.x_pred]; %#ok<AGROW>
    end
end
end

function plotMeanComparison(ax,baseline,corrected,localOnly)
if localOnly
    baselinePoints = baseline.points(baseline.points(:,1) >= 9,:);
    correctedPoints = corrected.points(corrected.points(:,1) >= 9,:);
    events = corrected.events(corrected.events(:,1) >= 10,:);
else
    baselinePoints = baseline.points;
    correctedPoints = corrected.points;
    events = zeros(0,7);
end

hold(ax,'on');
baseLine = plot(ax,baselinePoints(:,2),baselinePoints(:,3),'--o', ...
    'Color',[0.35,0.35,0.35],'LineWidth',1.0,'MarkerSize',3, ...
    'DisplayName','original CMA-ES mean');
correctedLine = plot(ax,correctedPoints(:,2),correctedPoints(:,3),'-o', ...
    'Color',[0,0.447,0.741],'LineWidth',1.25,'MarkerSize',3, ...
    'DisplayName','actual blended mean');

if ~isempty(events)
    normalPoints = plot(ax,events(:,2),events(:,3),'x', ...
        'Color',[0.85,0.325,0.098],'LineWidth',1.0,'MarkerSize',5, ...
        'DisplayName','normal CMA mean');
    predictionPoints = plot(ax,events(:,6),events(:,7),'d', ...
        'Color',[0.494,0.184,0.556],'LineWidth',1.0,'MarkerSize',4, ...
        'DisplayName','quadratic prediction');
    correctionVectors = quiver(ax,events(:,2),events(:,3),events(:,4)-events(:,2), ...
        events(:,5)-events(:,3),0,'Color',[0.466,0.674,0.188], ...
        'LineWidth',0.6,'MaxHeadSize',0.35, ...
        'DisplayName','applied correction');
    text(ax,events(:,4),events(:,5),compose('  g%d',events(:,1)), ...
        'FontSize',7,'Color',[0,0.447,0.741],'Clipping','on');
else
    normalPoints = plot(ax,NaN,NaN,'x','Color',[0.85,0.325,0.098], ...
        'DisplayName','normal CMA mean');
    predictionPoints = plot(ax,NaN,NaN,'d','Color',[0.494,0.184,0.556], ...
        'DisplayName','quadratic prediction');
    correctionVectors = quiver(ax,NaN,NaN,NaN,NaN,0, ...
        'Color',[0.466,0.674,0.188],'LineWidth',0.6, ...
        'DisplayName','applied correction');
end

eventCoordinates = [events(:,2:3); events(:,4:5); events(:,6:7)];
allCoordinates = [baselinePoints(:,2:3); correctedPoints(:,2:3); ...
    eventCoordinates];
setAxesLimits(ax,allCoordinates);
axis(ax,'equal');
grid(ax,'on');
box(ax,'on');
xlabel(ax,'m_1');
ylabel(ax,'m_2');
set(ax,'FontSize',9);
uistack(baseLine,'bottom');
uistack(correctedLine,'top');
uistack(normalPoints,'top');
uistack(predictionPoints,'top');
uistack(correctionVectors,'top');
hold(ax,'off');
end

function setAxesLimits(ax,coordinates)
coordinates = coordinates(all(isfinite(coordinates),2),:);
if isempty(coordinates)
    return;
end
xLimits = [min(coordinates(:,1)),max(coordinates(:,1))];
yLimits = [min(coordinates(:,2)),max(coordinates(:,2))];
xPadding = max(1e-4,0.06*max(diff(xLimits),1e-4));
yPadding = max(1e-4,0.06*max(diff(yLimits),1e-4));
xlim(ax,xLimits + [-xPadding,xPadding]);
ylim(ax,yLimits + [-yPadding,yPadding]);
end
