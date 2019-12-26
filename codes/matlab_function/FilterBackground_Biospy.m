% filter the white background in needle biospy slide
% Input: needle biospy slide rgb_img and the small area you want to filter
% Output: the binary mask and the tissue without white background
function [mask, tissue] = FilterBackground_Biospy(rgb_img, w, h)
    T = graythresh(rgb_img);
    bw = im2bw(rgb_img, T);  % the origin T is 0.75
    bw = imfill(~bw, 'holes');
   
    se = strel('disk', 15);
    bw1 = imopen(bw, se);    
    bw2 = imclose(bw1, se);
    bw3 = imfill(bw2, 'holes');
    bw3 = bwareaopen(bw3, w*h*3.14/2, 8); % 200*200 is the size of patch, the circle should contain it.
% overlap the mask on original image
    mask = cat(3, bw3, bw3, bw3);
    mask = uint8(mask);
    tissue = mask.*rgb_img;
    mask = im2bw(mask*255);
    disp('Filter background from biopsy Success...')
end
