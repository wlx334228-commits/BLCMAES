function files = plotContinuousScaleBranchResults(branchSummary,snapshotTable,figureRoot)
%PLOTCONTINUOUSSCALEBRANCHRESULTS Static, read-only diagnostic plots.
if exist(figureRoot,'dir') ~= 7, mkdir(figureRoot); end
files = cell(0,1);
if isempty(branchSummary), return; end
pred = branchSummary(~strcmp(branchSummary.eta_label,'A1'),:);
for smdNo = unique(pred.SMD)'
    rows = pred(pred.SMD == smdNo,:);
    figure('Visible','off','Color','w');
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    nexttile; hold on;
    for eta = unique(rows.eta)'
        part = rows(rows.eta == eta,:);
        scatter(part.d,part.auxiliary_improvement_over_start,36, ...
            'filled','DisplayName',sprintf('eta=%.2f',eta));
    end
    xlabel('standardized prediction offset d');
    ylabel('auxiliary-only improvement over branch-start best UF');
    title(sprintf('SMD%d: auxiliary benefit',smdNo)); legend('Location','best'); grid on;
    nexttile; hold on;
    for eta = unique(rows.eta)'
        part = rows(rows.eta == eta,:);
        scatter(part.q,part.overall_history_improvement,36, ...
            'filled','DisplayName',sprintf('eta=%.2f',eta));
    end
    xlabel('normalized local curvature q');
    ylabel('overall historical-best improvement');
    title(sprintf('SMD%d: overall branch benefit',smdNo)); legend('Location','best'); grid on;
    fileName = fullfile(figureRoot,sprintf('SMD%d_state_scale_benefit.png',smdNo));
    exportgraphics(gcf,fileName,'Resolution',160); close(gcf);
    files{end+1,1} = fileName; %#ok<AGROW>

    % State groups are descriptive only.  Each plotted point is first
    % averaged within a main run (over selected states and repeats), then
    % summarized across runs; no candidate/generation is counted as an
    % independent observation.
    dCut = median(rows.d,'omitnan'); qCut = median(rows.q,'omitnan');
    figure('Visible','off','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    plotGroupedCurve(rows,rows.d <= dCut, ...
        'auxiliary_improvement_over_start','d low','d high', ...
        'Auxiliary improvement grouped by d');
    plotGroupedCurve(rows,rows.d <= dCut, ...
        'overall_history_improvement','d low','d high', ...
        'Overall improvement grouped by d');
    plotGroupedCurve(rows,rows.q <= qCut, ...
        'auxiliary_improvement_over_start','q low','q high', ...
        'Auxiliary improvement grouped by q');
    plotGroupedCurve(rows,rows.q <= qCut, ...
        'overall_history_improvement','q low','q high', ...
        'Overall improvement grouped by q');
    fileName = fullfile(figureRoot,sprintf( ...
        'SMD%d_state_group_scale_curves.png',smdNo));
    exportgraphics(gcf,fileName,'Resolution',160); close(gcf);
    files{end+1,1} = fileName; %#ok<AGROW>
end
% Preserve a visible record of all selected state positions without treating
% phase as a causal effect.
if ~isempty(snapshotTable)
    figure('Visible','off','Color','w');
    scatter(snapshotTable.d,snapshotTable.q,40,double(snapshotTable.selected),'filled');
    xlabel('d'); ylabel('q'); title('Captured continuous-scale states'); grid on;
    fileName = fullfile(figureRoot,'state_coverage.png');
    exportgraphics(gcf,fileName,'Resolution',160); close(gcf);
    files{end+1,1} = fileName; %#ok<AGROW>
end

function plotGroupedCurve(rows,lowMask,metric,lowLabel,highLabel,plotTitle)
nexttile; hold on;
etas = unique(rows.eta)';
plotOneGroup(rows,lowMask,etas,metric,lowLabel,[0.20 0.45 0.75]);
plotOneGroup(rows,~lowMask,etas,metric,highLabel,[0.85 0.33 0.10]);
xlabel('auxiliary sampling scale \eta'); ylabel(strrep(metric,'_','\_'));
title(plotTitle); grid on; legend('Location','best');
end

function plotOneGroup(rows,groupMask,etas,metric,label,colour)
centre = NaN(size(etas)); lower = NaN(size(etas)); upper = NaN(size(etas));
for etaIndex = 1:numel(etas)
    part = rows(groupMask & rows.eta == etas(etaIndex),:);
    runValues = [];
    for runNo = unique(part.run_no)'
        values = part.(metric)(part.run_no == runNo);
        runValues(end+1,1) = mean(values,'omitnan'); %#ok<AGROW>
    end
    runValues = runValues(isfinite(runValues));
    if ~isempty(runValues)
        centre(etaIndex) = median(runValues);
        lower(etaIndex) = centre(etaIndex)-prctile(runValues,25);
        upper(etaIndex) = prctile(runValues,75)-centre(etaIndex);
    end
end
errorbar(etas,centre,lower,upper,'-o','Color',colour, ...
    'MarkerFaceColor',colour,'DisplayName',label);
end
end
