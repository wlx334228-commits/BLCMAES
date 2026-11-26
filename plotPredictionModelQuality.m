function files = plotPredictionModelQuality(analysis,outputRoot)
%PLOTPREDICTIONMODELQUALITY Run-level rates and stage summaries, PNG + FIG.
directory=fullfile(outputRoot,'figures');
if ~isfolder(directory), mkdir(directory); end
S=analysis.summary; stages=analysis.stageSummary;
ids=["QP","CP","RBF","GPR"]; smds=unique(S.SMD)'; files=cell(0,1);
metrics={'valid_prediction_rate_mean','effective_best_improvement_rate_mean', ...
    'conditional_best_improvement_rate_mean','conditional_parent_rate_mean'};
labels={'Valid prediction rate','Beats normal best / all opportunities', ...
    'Beats normal best / valid predictions','Potential parent / valid predictions'};
for metricIndex=1:numel(metrics)
    f=figure('Visible','off','Position',[100 100 1350 650],'Color','white');
    tiledlayout(2,4,'TileSpacing','compact');
    for smd=smds
        nexttile; values=NaN(1,4);
        for i=1:4
            index=S.SMD==smd&S.model==ids(i); values(i)=S.(metrics{metricIndex})(index);
        end
        bar(values); ylim([0 1]); xticks(1:4); xticklabels(ids);
        title(sprintf('SMD%d',smd)); ylabel(labels{metricIndex}); grid on;
    end
    stem=fullfile(directory,metrics{metricIndex});
    exportgraphics(f,[stem '.png'],'Resolution',150); savefig(f,[stem '.fig']);
    close(f); files{end+1,1}=[stem '.png']; %#ok<AGROW>
end
f=figure('Visible','off','Position',[100 100 1350 650],'Color','white');
tiledlayout(2,4,'TileSpacing','compact');
for smd=smds
    nexttile; hold on;
    for i=1:4
        y=NaN(1,3);
        for stage=1:3
            index=stages.SMD==smd&stages.model==ids(i)&stages.stage==stage;
            if any(index), y(stage)=stages.effective_best_improvement_rate_mean(index); end
        end
        plot(1:3,y,'-o','DisplayName',ids(i));
    end
    xticks(1:3); xticklabels({'Early','Middle','Late'}); ylim([0 1]);
    title(sprintf('SMD%d',smd)); ylabel('Beats normal best / all opportunities'); grid on;
    if smd==smds(1), legend('Location','best'); end
end
stem=fullfile(directory,'model_quality_by_stage');
exportgraphics(f,[stem '.png'],'Resolution',150); savefig(f,[stem '.fig']);
close(f); files{end+1,1}=[stem '.png'];
end
