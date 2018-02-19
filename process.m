clear,clc,close all

obj_list = object_list();
obj = '---'; % ALL
% obj = obj_list{5};
features = {'ecsad', 'ppfhist', 'ppfhistfull', 'shot', 'si', 'usc'};
rad='0.2';
inlier_threshold=10;

feat_name_map = containers.Map(...
    features,...
    {'ECSAD', 'PPFHist', 'PPFHistFull', 'SHOT', 'SpinImage', 'USC'}...
);


%% Process
hold on;
legends = cell(1,length(features));
for idx=1:length(features)
    feat = features{idx};

    %% Load
    flist = dir(['output/' feat rad '*' obj '*']);
    results = cell(length(flist),1);
    fprintf('Loading %i data files for feature %s...\n', length(flist), feat_name_map(feat))
    for i=1:length(flist)
        fi = [flist(i).folder '/' flist(i).name];
        fid = fopen(fi, 'r');
        results{i} = reshape(fread(fid, inf, 'double'), 3, [])';
        fclose(fid);
    end
    results = cell2mat(results);
    
    
    %% Compute stats
    num_positives = sum(results(:,3));
    retrieved = results( ~isnan(results(:,2)) , 1:2 );
    num_retrieved = size(retrieved,1);
    inliers = results( results(:,1)<inlier_threshold & ~isnan(results(:,2)) , 1:2 );
    
    
    %% Compute PR
    [matching_distance_sort,order] = sort(retrieved(:,2), 'ascend');
    alignment_distance_sort = retrieved(order,1);
    alignment_distance_mask = (alignment_distance_sort < inlier_threshold);
    precision = cumsum(alignment_distance_mask) ./ (1:num_retrieved)';
    recall = cumsum(alignment_distance_mask) / num_positives;
    % auc = trapz(1-precision, recall);
    auc = trapz(recall, precision);
    max_f1 = max(2 * precision .* recall ./ (precision + recall));
    
    %% Plot
%     plot(1-precision, recall)
    plot(recall, precision)
    legends{idx} = sprintf('%s (AUC %.3f, F1 %.3f)', feat_name_map(feat), auc, max_f1);
end

% xlim([0,1])
% ylim([0, 1])
% xlabel('1 - Precision')
% ylabel('Recall')
% legend(legends, 'Location', 'NorthWest')

xlim([0, 0.25])
ylim([0, 1])
xlabel('Recall')
ylabel('Precision')
legend(legends, 'Location', 'NorthEast')
hold off