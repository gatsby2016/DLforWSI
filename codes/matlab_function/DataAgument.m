% Data Augment for images. 
% input: the origin image
% output: the image after rotating, 0 90 180  270,  up_down mirror, left_right mirror

function DataAgument(img, savepath, shortname,number)
    if number >= 1
        img1 = img;
        imwrite(img1, [savepath, shortname, '_1.png']);
    end
    if number >= 2
        img2 = imrotate(img,90); 
        imwrite(img2, [savepath, shortname,  '_2.png']);
%     figure(2);
%     imshow(img2);  
    end
    if number >=3
        img3 = imrotate(img,180);
        imwrite(img3,[savepath, shortname,  '_3.png']);
%     figure(3);
%     imshow(img3);
    end
    if number >=4
        img4 = imrotate(img,270);
        imwrite(img4,[savepath, shortname,  '_4.png']);
%     figure(4);
%     imshow(img4);
    end
    if number >=5
        img5 = img(end:-1:1,:,:); % up-down mirror 
        imwrite(img5,[savepath, shortname,  '_5.png']);
%     figure(5);
%     imshow(img5);
    end
    if number >= 6
        img6 = img(:,end:-1:1,:); % left-right mirror
        imwrite(img6,[savepath, shortname,  '_6.png']);
%     figure(6);
%     imshow(img6);
    end
end