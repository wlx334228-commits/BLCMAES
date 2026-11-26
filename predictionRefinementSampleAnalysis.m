function row=predictionRefinementSampleAnalysis(event,prepared,smd,runNo,seed,index)
%PREDICTIONREFINEMENTSAMPLEANALYSIS Descriptive coverage, never selection gates.
H1=prepared.H1.selectedSamples; H2=zeros(0,5);
if prepared.P1valid, H2=prepared.H2.selectedSamples; end
first=max(1,event.generation-9);
eligible=event.history(event.history(:,1)>=first&event.history(:,1)<=event.generation,:);
row=struct('SMD',smd,'run_no',runNo,'seed',seed,'state_index',index, ...
    'generation',event.generation,'main_UFE',event.UFE, ...
    'P1_valid',prepared.P1valid,'P1_reason',string(prepared.P1reason), ...
    'full_archive_count',size(event.history,1),'eligible_archive_count',size(eligible,1), ...
    'first_eligible_generation',first,'sample_count_H1',size(H1,1),'sample_count_H2',size(H2,1), ...
    'H1_indices',string(jsonencode(H1(:,1:2))),'H2_indices',string(jsonencode(H2(:,1:2))), ...
    'H1_samples',string(jsonencode(H1)),'H2_samples',string(jsonencode(H2)), ...
    'H2_source',"full true archive, same ten-generation eligibility; not H1", ...
    'overlap_count',0,'overlap_rate',NaN,'H2_new_sample_count',0);
if ~isempty(H2)
    row.overlap_count=size(intersect(H1(:,1:2),H2(:,1:2),'rows'),1);
    row.overlap_rate=row.overlap_count/size(H1,1);
    row.H2_new_sample_count=size(H2,1)-row.overlap_count;
end
sets={H1,H2}; centres={event.CMA.xmean(1:2),prepared.models(1).x};
for i=1:2
    prefix=sprintf('H%d_',i); H=sets{i}; centre=centres{i};
    names={'distance_min','distance_median','distance_max','UF_min','UF_q25', ...
        'UF_median','UF_mean','UF_std','UF_q75','UF_max','age_min','age_median','age_max', ...
        'spread_eigen_min','spread_eigen_max','spread_condition','hull_area', ...
        'centre_inside_hull','largest_angular_gap','quadratic_design_condition'};
    for name=names, row.([prefix name{1}])=NaN; end
    row.([prefix 'distances'])=""; row.([prefix 'spread_covariance'])="";
    if isempty(H)||any(~isfinite(centre)), continue; end
    offsets=H(:,3:4)-centre; distances=vecnorm(offsets,2,2);
    row.([prefix 'distances'])=string(jsonencode(distances));
    row.([prefix 'distance_min'])=min(distances);
    row.([prefix 'distance_median'])=median(distances);
    row.([prefix 'distance_max'])=max(distances);
    UF=H(:,5); qs=prctile(UF,[25 50 75]);
    row.([prefix 'UF_min'])=min(UF); row.([prefix 'UF_max'])=max(UF);
    row.([prefix 'UF_q25'])=qs(1); row.([prefix 'UF_median'])=qs(2);
    row.([prefix 'UF_q75'])=qs(3); row.([prefix 'UF_mean'])=mean(UF);
    row.([prefix 'UF_std'])=std(UF);
    ages=event.generation-H(:,1); row.([prefix 'age_min'])=min(ages);
    row.([prefix 'age_median'])=median(ages); row.([prefix 'age_max'])=max(ages);
    z=offsets/max(max(distances),realmin); spread=cov(z,1); ev=eig(spread);
    row.([prefix 'spread_covariance'])=string(jsonencode(spread));
    row.([prefix 'spread_eigen_min'])=min(ev); row.([prefix 'spread_eigen_max'])=max(ev);
    row.([prefix 'spread_condition'])=cond(spread);
    design=[ones(size(z,1),1),z,z(:,1).^2,z(:,1).*z(:,2),z(:,2).^2];
    row.([prefix 'quadratic_design_condition'])=cond(design);
    angles=sort(atan2(offsets(distances>0,2),offsets(distances>0,1)));
    if ~isempty(angles), row.([prefix 'largest_angular_gap'])=max(diff([angles;angles(1)+2*pi])); end
    if rank(z)>=2
        [hull,area]=convhull(H(:,3),H(:,4));
        row.([prefix 'hull_area'])=area;
        row.([prefix 'centre_inside_hull'])=double(inpolygon(centre(1),centre(2),H(hull,3),H(hull,4)));
    end
end
end
