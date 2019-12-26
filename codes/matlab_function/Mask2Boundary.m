% convert bw mask to boundary. show the countour on rgb-image.
% input: the rgb image and bw mask;
% input: boundary_color, 'r', 'g', 'b'
% ouput: the cell of Boundary, and show the coutour on the rgb-mask.
function Boundary = Mask2Boundary(img, bw, boundary_color)
    imshow(img);
    hold on;
    
    Boundary = bwboundaries(bw);
    for i=1:length(Boundary)
        curB=Boundary{i};
        plot(curB(:,2),curB(:,1), boundary_color, 'LineWidth',3);
    end
%     title(['The object boundary num is: ', num2str(i)])
    %     axis('image'); axis('off')
end
