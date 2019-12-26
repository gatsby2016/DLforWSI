%% for ID 51554 visualization, All Tissue Patches saving
clear;
clc;
tic;
% addpath('D:\Matlab2018b\downloads\openslide-3.4.1\')
addpath('D:\Matlab2018b\downloads\openslide-matlab\')
addpath('matlab_function\');
% addpath('chomp_function'); % openslide_load_library
disp('Functions Loading Success...')

% @0 read the slide images
path = 'E:\Data\CCRCC_KidneyFCH\';
allfile = dir([path, '*xml']);
filenames  = {allfile.name};

%% @1 parameters setting
height = 512;  
width = 512;
step = 512;
keepr = 0.75; %% drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
format = '.png';
scale = 1;
savepath = 'E:\Projects\ccRCCTumor\data\train\non\';
linecolorvalue = 255;
disp('Parameters Setting Success...')

%%
for num = 1:length(filenames)
    namesplit = strsplit(filenames{num}, '.');
    casename = [namesplit{1}, '.mrxs'];
    disp([num2str(num), ' / ', num2str(length(filenames)), '  ------ ',  casename, ' checked!'])

%     svs = openslide_open();
    DataPreparationV2([path,casename], linecolorvalue, height, width, step, scale,  keepr, savepath, format);
end
toc;
%% Supplementary: read the whole image at 10x from svs file
% levels = openslide_get_level_count(svs); % nums is 11, 0 is 40x, 1 is 10x, 2 is 2.5x
% [w,h] = openslide_get_level_dimensions(svs, 0);
% img= openslide_read_region(svs, w/2, h/4, 5000, 5000, 0); % read the whole image at 10x
% img = img(:, :, 2:end); % variable 'img' is the case image
% disp('The Whole Image Loading from SVS Success...')
%% @3 proprecessing the whole image£¬ getting the tissue part mask
% [mask, ~] = FilterBackground_Biospy(img, height, width); 
% proprecessing the image, filter the background
% mask(1:3200,1:3200) = 0; % filter the blue mirror part
% imshow(imresize(mask, 0.05))
%% %%%%%%%%%%%%% @Optional, show the boundary on image
% Mask2Boundary(img, mask, 'g');
%% @4 getting patches from the whole image
% for i = 1: step: size(img, 1) - height  % height 
%     for j = 1: step: size(img, 2) - width % width
%         region = mask(i : i + height - 1, j : j + width - 1);
%         if sum(sum(region)) < height*width*drop_rate
%             continue;
%         end
        %% %%%%%%%% @optional, show the patch boundarybox on image and mask
%         linesy = cat(2, ones(1,width)*i, [i: i+height], ones(1,width)*(i+height), [i+height:-1:i]);
%         linesx = cat(2, [j:j+width], ones(1,height)*(j+width), [j+width:-1:j], ones(1,height)*j);
%         plot(linesx,linesy, 'b', 'LineWidth',1);
        %% %%%%%%%%           
%         patch = img(i : i + height - 1, j : j + width - 1, :);
% %         imshow(patch); pause;
%         patch_name = [savepath, num2str((i-1)/step+1),'_', num2str((j-1)/step+1), format];
%         imwrite(patch, patch_name);
%     end
% end
% disp(['Patches finished! Folder: ', savepath]);
% prediction = zeros((i-1)/step+1, (j-1)/step+1);
% save MaskPredictionReserved.mat prediction
% imshow(prediction)
%%
toc;