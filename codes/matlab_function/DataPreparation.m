%% 8 inputs parameters function for data preparation from svs at the highest scale
%% input:  'path' of  svs files;   eg. '/home/yann/data/train/'
%% input:  the h/w/step of patch you want to get;   eg.  200, 200,100
%% input:  linecolorvalue is the label line color; eg. '16711680'-blue; '65280'-green; '255'-red
%% input: scale is the resolution, 1 is 40x, 2 is 20x, 4 is 10x, 8 is 5x
%% input:  drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
%% input:  savepath is the path to save the image patches
%% input: format, the output images format , such as '.png' or '.bmp'
%% input:  If_show is a bool value whether show the patch or not;  eg. 'True'
%%
function DataPreparation(path, h, w, step, linecolorvalue, scale,  drop_rate, savepath, format, If_Show)
    addpath('D:\Matlab2018b\downloads\openslide-matlab\')
%     path = '/media/yann/FILE/MI___Data/GLH/test/4+4/';
    if path(end-3) == '.'
        signindex = strfind(path, '\');
        svs_name = {path(signindex(end)+1:end)};
    else
        Allfile = dir([path, '*.svs']);
        svs_name = {Allfile.name};
    end
   %% parameters
%     h = 200; w = 200; step = 100; 
%     linecolorvalue = 16711680; %G3:16711680   G4:65280  BENIGN: 255;  different color represents different grade
    factor = 10; % sampling factor is 4 just a factor in case of Out Of Memory
%     drop_rate = 0.75;
%     savepath = '/home/yann/Projects/GLH_data_classification/data/test/g3/';

    %% loop for each slide
    for n  = 1: 1: length(svs_name)
        count = 0;
        name = svs_name{n};
        id = name(1:end-4);
        disp(['****************** ', num2str(n), ' / ', num2str(length(svs_name)),  '    ',  num2str(id), ' ******************' ])
        
        if path(end-3) == '.'
            [color, annotation_info] = GetAnnotation_MultiColor_XML([path(1:end-4), '.xml']); % get the struct of the xml annotation
        else
            [color, annotation_info] = GetAnnotation_MultiColor_XML([path, id, '.xml']); % get the struct of the xml annotation
        end
        
        if ~ismember(linecolorvalue, color) % if 'linecolorvalue' are not in 'color', then 'continue'
            continue;
        end
        
      %% read the image slides by openslide
        if path(end-3) == '.'
            svs = openslide_open(path);
        else
            svs = openslide_open([path, id, '.svs']);
        end
        [width, height] = openslide_get_level_dimensions(svs, 0);

      %% process the annotation.
        index = find([annotation_info.linecolor] == linecolorvalue);
        position = {annotation_info(index).X; annotation_info(index).Y}';    

      %% loop for each ROI region in a slide 
        for ind = 1: 1: size(position,1)
            disp(['Now is ROI, ', num2str(ind)]);
            P = [position{ind,1}, position{ind, 2}];
            PP = [P; P(1,:)];
            MASK = poly2mask(PP(:,1)/factor, PP(:,2)/factor, double(height/factor), double(width/factor));
%             imshow(MASK)  %........................................................imshow
            Box = regionprops(MASK,'BoundingBox');
            widthstart = int64(Box.BoundingBox(1));
            widthlen  = int64(Box.BoundingBox(3));
            heightstart = int64(Box.BoundingBox(2));
            heightlen = int64(Box.BoundingBox(4));
            BoundingMask = imresize(MASK(heightstart: heightstart+heightlen, widthstart: widthstart+widthlen), factor);
    %         imshow(BoundingMask)

        %% slide window for getting patches
           for i = 1: step*scale: size(BoundingMask,1) - h*scale + 1
                for j = 1: step*scale: size(BoundingMask,2) - w*scale + 1
                    region = BoundingMask(i: i+h*scale-1, j: j+w*scale-1);

                    if sum(sum(region)) < h*w*scale*scale*drop_rate
                        continue;
                    end

                    patch  = openslide_read_region(svs, widthstart * factor + j - 1, heightstart * factor + i - 1,  w*scale, h*scale, 0);
                    patch = patch(:, :, 2:end);
                    patch = imresize(patch, [w, h]);

                    count = count+1;
                    patch_name = [savepath, id, '_', num2str(count), format];
%                     disp(patch_name)
                    if If_Show
                        imshow(patch)
                        pause;
                    else
                        imwrite(patch, patch_name);
                    end
                end
            end
        end
    end
end
%%