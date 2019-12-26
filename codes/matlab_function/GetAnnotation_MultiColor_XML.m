%% GetAnnotation from xmlfile by different color annotation
% input: xml file
% output: struct including annotation info
%  chaoyang, 2018.10.10

function [color, annotation_info] = GetAnnotation_MultiColor_XML(xmlFile)
    color = [];
    xDoc = xmlread(xmlFile);
    Annotation = xDoc.getElementsByTagName('Annotation'); % read xmlfile and get all annotations

    init = 0; % for multi-color U
    annotation_info = [];       
    for l = 0: 1: Annotation.getLength - 1 % loop all different linecolor Annotation
        item_Annotation = Annotation.item(l);
        linecolor = item_Annotation.getAttribute('LineColor');
        Region = item_Annotation.getElementsByTagName('Region'); % get the tag, 'Region' in this item_Annotation
      
        for i = 0: 1: Region.getLength-1
            item_Region = Region.item(i);
            % grab ROI ID  area and Length
            ROI_id      = cell(item_Region.getAttribute('Id'));      
            ROI_id    = str2double(ROI_id{1});   
            ROI_area    = cell(item_Region.getAttribute('Area'));  
            ROI_area    = str2double(ROI_area{1});           
            ROI_Length    = cell(item_Region.getAttribute('Length'));  
            ROI_Length    = str2double(ROI_Length{1});
          
            % find child containing vertices for current ROI
            Vertices = item_Region.getFirstChild;
            while ~strcmpi(Vertices.getNodeName,'Vertices')
                Vertices = Vertices.getNextSibling;
            end
         
            % get vertices for current ROI
            X=[]; Y=[];
            Vertex = Vertices.getFirstChild;
            while ~isempty(Vertex)
                if strcmpi(Vertex.getNodeName,'Vertex')
                    x = cell(Vertex.getAttribute('X')); 
                    x = str2double(x{1});
                    y = cell(Vertex.getAttribute('Y')); 
                    y = str2double(y{1});
                    X = [X; x];
                    Y = [Y; y];
                end
                Vertex = Vertex.getNextSibling;
            end
            
            % write these info to 'annotation_info'
            pos = init+i+1;
            annotation_info(pos).roi_id  = ROI_id; %{1};        
            annotation_info(pos).linecolor  = str2num(linecolor); 
            annotation_info(pos).area    = ROI_area; %{1};
            annotation_info(pos).Length    = ROI_Length; %{1};
            annotation_info(pos).X       = X;
            annotation_info(pos).Y       = Y;
        end
        init = pos;
        disp([num2str(l+1), '  Num of ROI: ', num2str(i+1),  '  linecolor is:  ', char(linecolor)])
        color = [color; str2num(linecolor)];
    end
end
%%