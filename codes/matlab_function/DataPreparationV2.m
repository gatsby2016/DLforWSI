%% Chaoyang 20191031 version2 changing for one slide.
%% 8 inputs parameters function for data preparation from svs at the highest scale
%% input:  'path' of  svs files;   eg. '/home/yann/data/train/'
%% input:  the h/w/step of patch you want to get;   eg.  200, 200,100
%% input:  linecolorvalue is the label line color; eg. '16711680'-blue; '65280'-green; '255'-red
%% input: scale is the resolution, 1 is 40x, 2 is 20x, 4 is 10x, 8 is 5x
%% input:  drop_rate; only white region area in a mask patch > w*h*drop_rate can be saved
%% input:  savepath is the path to save the image patches
%% input: format, the output images format , such as '.png' or '.bmp'
%% input:  If_show is a bool value whether show the patch or not;  eg. 'True'

function DataPreparationV2(imgpath, linecolorvalue, h, w, step, scale,  drop_rate, savepath, format)
    factor = 10; % sampling factor is 4 just a factor in case of Out Of Memory
    
    splitID = strsplit(imgpath, {'\', '.', ' '});
    id = splitID{end-2};
    %% process the annotation.
    splitpart = strsplit(imgpath, '.');
    xmlpath= [splitpart{1}, '.xml'];
    [color, annotation_info] = GetAnnotation_MultiColor_XML(xmlpath); % get the struct of the xml annotation
    if ~ismember(linecolorvalue, color) % if 'linecolorvalue' are not in 'color', then 'continue'
        error([num2str(linecolorvalue), ' is not in XML! Please check the color coder!'])
        return;
    end
    index = find([annotation_info.linecolor] == linecolorvalue);
    position = {annotation_info(index).X; annotation_info(index).Y}';    
    
    %% read the image slides by openslide
    pointer = openslide_open(imgpath);
    [width, height] = openslide_get_level_dimensions(pointer, 0);
    count = 0;
    
  %% loop for each ROI region in a slide 
    for ind = 1: 1: size(position,1)
        disp(['Now is ROI, ', num2str(ind)]);
        P = [position{ind,1}, position{ind, 2}];
        PP = [P; P(1,:)];
        MASK = poly2mask(PP(:,1)/factor, PP(:,2)/factor, double(height/factor), double(width/factor));
%         imshow(MASK)  %........................................................imshow
%         hold on;
%         plot(1516.5+3,5169.5+3, 'r*')
        Box = regionprops(MASK, 'BoundingBox');
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

                patch  = openslide_read_region(pointer, widthstart * factor + j - 1, heightstart * factor + i - 1,  w*scale, h*scale, 0);
                patch = patch(:, :, 2:end);
                patch = imresize(patch, [w, h]);

                count = count+1;
%                     disp(patch_name)
%                     imshow(patch)
%                     pause;
                imwrite(patch,  [savepath, id, '_', num2str(count), format]);
            end
        end
    end
end
%%