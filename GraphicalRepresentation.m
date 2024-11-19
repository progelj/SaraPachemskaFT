connectivityMatrixCNN = [
    NaN, 0.8473, 0.6733, 0.7997, 0.9132, 0.8055, 0.8624, 0.7808, 0.7813, 0.7801, 0.9063, 0.8746, 1.0978, 0.7726, 0.8472, 0.8331, 0.7279, 0.7927, 0.6129;
    0.5929, NaN, 0.4944, 0.4529, 0.6195, 0.5200, 0.5351, 0.5517, 0.5940, 0.5422, 0.6078, 0.5229, 0.5495, 0.6034, 0.6438, 0.4752, 0.4334, 0.5355, 0.2397;
    0.6889, 0.9960, NaN, 0.8378, 0.6661, 0.7208, 0.6018, 0.6407, 0.5731, 0.8099, 0.6896, 0.9933, 0.6768, 0.6412, 0.7017, 0.8950, 0.7134, 0.8476, 0.7788;
    0.6737, 0.6066, 0.1911, NaN, 0.3333, 0.4308, 0.3953, 0.7795, 0.5491, 0.2108, 0.2938, 0.3623, 0.4682, 0.5041, 0.8019, 1.1341, 0.4172, 0.8840, 0.5867;
    0.3538, 0.3616, 1.1925, 0.4386, NaN, 1.1008, 0.4550, 0.9277, 0.5094, 1.4140, 0.1118, 0.3755, 1.2367, 1.1431, 0.5833, 0.5325, 1.2515, 0.4707, 0.8136;
    0.6551, 0.2737, 0.5792, 0.5677, 0.4516, NaN, 0.3445, 0.4078, 0.5664, 0.2637, 0.7402, 0.6808, 0.3578, 0.4411, 0.3829, 0.7059, 0.5179, 0.6732, 0.4341;
    0.7259, 0.7479, 0.5275, 0.9356, 0.8362, 0.6803, NaN, 1.0764, 1.6690, 0.5435, 0.5694, 0.6994, 0.8922, 0.5001, 0.6008, 0.9618, 0.9393, 1.0745, 0.9239;
    0.8517, 1.0261, 0.5987, 0.7251, 0.9451, 0.7778, 1.6925, NaN, 1.4232, 0.7373, 0.6335, 0.5909, 0.9293, 0.8352, 0.6528, 1.0480, 0.8422, 0.8247, 1.0573;
    0.7475, 0.4938, 0.4766, 0.7409, 0.8747, 0.7144, 1.8683, 1.5273, NaN, 0.4005, 0.6987, 0.5754, 0.8585, 0.8818, 0.9963, 0.8709, 1.0685, 0.9348, 0.9087;
    0.5089, 0.6305, 0.3581, 0.6091, 0.6708, 0.4889, 0.4271, 0.5876, 0.5218, NaN, 0.7484, 0.7475, 0.8186, 0.7132, 0.5833, 0.3337, 0.6170, 0.5047, 0.4953;
    0.3866, 0.3345, 0.3230, 0.3549, 0.3622, 0.5748, 0.3883, 0.3258, 0.3285, 0.4648, NaN, 0.4997, 0.3188, 0.2509, 0.4527, 0.3345, 0.3454, 0.3741, 0.3785;
    0.5187, 0.5418, 0.5753, 0.4573, 0.5511, 0.6112, 0.6452, 0.6859, 0.4982, 0.7070, 0.5037, NaN, 0.6384, 05297, 0.5036,  0.6106, 0.5908, 0.6021, 0.5438;
    0.5545, 0.7169, 0.7532, 0.6135, 0.6958, 0.6344, 0.8723, 0.6886, 0.7378, 0.6755, 0.6277, 0.5614, NaN, 0.9663, 0.6140, 0.8000, 0.6738, 0.9934, 0.6781;
    0.4757, 0.4664, 0.7321, 0.7762, 0.8285, 0.6122, 0.9532, 0.7518, 0.8524, 0.9010, 0.7398, 0.6192, 0.8210, NaN, 0.9761, 0.3980, 0.6778, 0.7919, 0.6905;
    0.7879, 0.6444, 0.5546, 0.9662, 0.9511, 0.8206, 0.6614, 0.8497, 0.8346, 0.6094, 0.6309, 0.6182, 0.6774, 0.9824, NaN, 0.9535, 0.8207, 0.6660, 0.7440;
    0.8569, 0.6439, 0.6611, 1.0024, 0.8224, 0.7447, 0.7867, 0.8371, 0.8110, 0.6938, 0.8038, 0.7108, 0.8266, 0.6877, 0.9682, NaN, 1.0411, 0.8317, 0.7398;
    0.8453, 0.7027, 0.5275, 0.6110, 0.6747, 0.5842, 0.6224, 0.7171, 0.7592, 0.4852, 0.5375, 0.5907, 0.6102, 0.5927, 0.4998, 0.8637, NaN, 0.9103, 0.6114;
    1.0062, 0.4343, 0.6124, 1.0724, 0.9685, 1.0610, 1.2134, 0.7532, 0.8526, 0.9292, 0.7500, 1.1282, 1.1701, 1.2402, 1.1134, 0.9520, 0.8911, NaN, 0.9554;
    0.6262, 0.5291, 0.5053, 0.9296, 0.8871, 0.5262, 0.9338, 0.8534, 0.9510, 0.6670, 0.5462, 0.5429, 0.5429, 0.5839, 0.7441, 0.9152, 0.5629, 1.0545, NaN;
    ];


% Replace NaN with 0 for graph representation
connectivityMatrixCNN(isnan(connectivityMatrixCNN)) = 0;

% Create a directed graph
G = digraph(connectivityMatrixCNN);

% Set the weights (edge thickness) based on matrix values
LWidths = 5 * G.Edges.Weight / max(G.Edges.Weight);

% Color mapping: Map edge weights to a color scale
colormap jet; % Set color map
edgeColors = G.Edges.Weight / max(G.Edges.Weight); % Normalize weights for color mapping

% Create a plot with customized layout
figure;
h = plot(G, ...
    'LineWidth', LWidths, ...
    'EdgeAlpha', 0.8, ...          % Transparency of edges
    'NodeColor', 'k', ...          % Black node color
    'MarkerSize', 8, ...           % Node size
    'ArrowSize', 10);              % Arrow size

% Set edge colors based on weights
edgeColorIndices = ceil(edgeColors * size(colormap, 1));
edgeColorIndices(edgeColorIndices == 0) = 1; % Avoid index 0
colors = colormap;
for i = 1:numedges(G)
    highlight(h, 'Edges', i, 'EdgeColor', colors(edgeColorIndices(i), :));
end

% Layout: Use circular layout for better readability
layout(h, 'circle');

% Add node labels for description
nodeLabels = arrayfun(@(x) sprintf('Node %d', x), 1:numnodes(G), 'UniformOutput', false);
labelnode(h, 1:numnodes(G), nodeLabels);

% Add title and legend
title('Creative Connectivity Graph - CNN', 'FontSize', 14, 'FontWeight', 'bold');

% Customize axis
axis off; % Hide axis for clean visualization
colorbar; % Show color scale for weights
% Replace NaN with 0 for graph representation
connectivityMatrixCNN(isnan(connectivityMatrixCNN)) = 0;

% Create a directed graph
G = digraph(connectivityMatrixCNN);

% Set the weights (edge thickness) based on matrix values
LWidths = 5 * G.Edges.Weight / max(G.Edges.Weight);

% Color mapping: Map edge weights to a color scale
colormap jet; % Set color map
edgeColors = G.Edges.Weight / max(G.Edges.Weight); % Normalize weights for color mapping

% Create a plot with customized layout
figure;
h = plot(G, ...
    'LineWidth', LWidths, ...
    'EdgeAlpha', 0.8, ...          % Transparency of edges
    'NodeColor', 'k', ...          % Black node color
    'MarkerSize', 8, ...           % Node size
    'ArrowSize', 10);              % Arrow size

% Set edge colors based on weights
edgeColorIndices = ceil(edgeColors * size(colormap, 1));
edgeColorIndices(edgeColorIndices == 0) = 1; % Avoid index 0
colors = colormap;
for i = 1:numedges(G)
    highlight(h, 'Edges', i, 'EdgeColor', colors(edgeColorIndices(i), :));
end

% Layout: Use circular layout for better readability
layout(h, 'circle');

% Add node labels for description
nodeLabels = arrayfun(@(x) sprintf('Node %d', x), 1:numnodes(G), 'UniformOutput', false);
labelnode(h, 1:numnodes(G), nodeLabels);

% Add title and legend
title('Creative Connectivity Graph - CNN', 'FontSize', 14, 'FontWeight', 'bold');

% Customize axis
axis off; % Hide axis for clean visualization
colorbar; % Show color scale for weights
