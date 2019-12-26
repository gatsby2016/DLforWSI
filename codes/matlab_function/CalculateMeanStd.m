% Calculate the mean and standard deviation for all images in 'path'. 
% input: the path and images format
% output: the mean and std of each channel

function [mean_channel, std_channel] = CalculateMeanStd(path, format)
    allfile = dir([path, format]);
    allname = {allfile.name};

    %%
    mean_channel_sum = 0.0;
    std_channel_sum = 0.0;
    
    for num = 1: 1: length(allname)
        name =allname{num};
        disp(name);
        img = imread([path, name]);
        img = double(img)/255;
        mean_channel_sum = mean_channel_sum + [mean2(img(:,:,1))  mean2(img(:,:,2))  mean2(img(:,:,3))];
        std_channel_sum      = std_channel_sum      + [std2(img(:,:,1))    std2(img(:,:,2))    std2(img(:,:,3))];
    end
    %%
    mean_channel = mean_channel_sum / num;   %length(allname);
    std_channel = std_channel_sum / num;   %length(allname);
end