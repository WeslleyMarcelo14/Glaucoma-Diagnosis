% Read RGB image
I = imread('FundusImages/RET002OD.jpg');
I = imresize(I, [256, 256]);

% (Optional) Convert to HSV to get the brightness channel V
HSV = rgb2hsv(I);
V = HSV(:,:,3);

% Show original image
figure;
imshow(I);
title('Original Image');

% ----- PREPARE FEATURES FOR DBSCAN -----
[rows, cols, ~] = size(I);

% Normalized color channels (0-1)
R = double(I(:,:,1)) / 255;
G = double(I(:,:,2)) / 255;
B = double(I(:,:,3)) / 255;

% Normalized spatial coordinates
[xGrid, yGrid] = meshgrid(1:cols, 1:rows);
xNorm = xGrid(:) / cols;
yNorm = yGrid(:) / rows;

% Color vectors as columns
Rvec = R(:);
Gvec = G(:);
Bvec = B(:);
Vvec = V(:);  % brightness (optional, but helps)

% Build feature vector: [color + position]
% You can test combinations, for example:
% features = [Rvec, Gvec, Bvec, xNorm, yNorm];
features = [Rvec, Gvec, Bvec, Vvec, xNorm, yNorm];

% Normalize features (greatly improves DBSCAN)
features = zscore(features);

% ----- APPLY DBSCAN -----
epsilon = 0.7;    % neighbor radius (adjust by trial and error)
minPts  = 50;     % minimum points per cluster (also adjust)

labels = dbscan(features, epsilon, minPts);

% labels = -1 are noise; 1,2,3,... are valid clusters

% ----- RECONSTRUCT SEGMENTED IMAGE (OPTIONAL, VISUALIZATION) -----
segmented_image = zeros(rows, cols, 3, 'uint8');

unique_labels = unique(labels);
unique_labels(unique_labels == -1) = [];  % remove noise

for i = 1:numel(unique_labels)
    lbl = unique_labels(i);
    mask_cluster = (labels == lbl);

    % average color of cluster for visualization
    Rmean = mean(Rvec(mask_cluster)) * 255;
    Gmean = mean(Gvec(mask_cluster)) * 255;
    Bmean = mean(Bvec(mask_cluster)) * 255;

    cluster_rgb = uint8(cat(3, ...
        Rmean * ones(rows, cols), ...
        Gmean * ones(rows, cols), ...
        Bmean * ones(rows, cols)));

    segmented_image(repmat(reshape(mask_cluster, rows, cols), [1 1 3])) = ...
        cluster_rgb(repmat(reshape(mask_cluster, rows, cols), [1 1 3]));
end

figure;
imshow(segmented_image);
title('DBSCAN Segmentation');

% ----- FIND BRIGHTEST CLUSTER (OPTIC DISC) -----
cluster_ids = unique_labels;
num_clusters = numel(cluster_ids);
cluster_brightness = zeros(num_clusters, 1);

for i = 1:num_clusters
    lbl = cluster_ids(i);
    mask_cluster = (labels == lbl);
    % use V (brightness in HSV) to measure
    cluster_brightness(i) = mean(Vvec(mask_cluster));
end

[~, idx_max] = max(cluster_brightness);
brightest_cluster = cluster_ids(idx_max);

% Create binary mask of the brightest cluster
mask = reshape(labels == brightest_cluster, rows, cols);

figure;
imshow(mask);
title('Optic Disc Mask (DBSCAN)');

% ----- DETECT EDGES WITH CANNY -----
edges = edge(mask, 'Canny');

figure;
imshow(edges);
title('Canny Edges (on Mask - DBSCAN)');

% ----- APPLY MASK TO ORIGINAL IMAGE -----
isolated_disc = I;
for c = 1:3
    channel = isolated_disc(:,:,c);
    channel(~mask) = 0;
    isolated_disc(:,:,c) = channel;
end

figure;
imshow(isolated_disc);
title('Isolated Optic Disc (DBSCAN)');
