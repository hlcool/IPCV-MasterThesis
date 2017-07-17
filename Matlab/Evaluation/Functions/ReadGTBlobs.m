function [Blobs, Blobs_track num_blobs_total]=ReadGTBlobs(annotationFileName,video_name,processing_mask)



fidIn = fopen(annotationFileName, 'r');
if fidIn <= 0
    error('Open file %s failed!', annotationFileName);
end;

object_number=0;
bbFirstFrame = -1;
num_blobs_total=0;
while ~feof(fidIn)
    tline=fgetl(fidIn);
    if strfind(tline,'Object Number') == 1
        object_number = sscanf(tline, 'Object Number #%d');
        bbGT{object_number} = [];
        bbFirstFrame = -1;
    end
    if strfind(tline,'End Frame') == 1
        endFrame = sscanf(tline, '%*s Frame:%d');
        if isempty(bbGT{object_number})
            bbGT{object_number} = zeros(endFrame+1, 4);
        else
            temp = zeros(endFrame+1, 4);
            temp(1:size(bbGT{object_number}, 1), :) = bbGT{object_number};
            bbGT{object_number} = temp;
        end;
    end;
    if strfind(tline,'Frame') == 1
        frameNo = sscanf(tline, '%*s %d');
        x = sscanf(fgetl(fidIn), '%*s %d');
        y = sscanf(fgetl(fidIn), '%*s %d');
        width = sscanf(fgetl(fidIn), '%*s %d');
        height = sscanf(fgetl(fidIn), '%*s %d');
        
        
            
        if width == 10 && height == 10
            x = 0; y = 0; width = 0; height = 0;
        end;
        if bbFirstFrame == -1
            bbFirstFrame = frameNo(1)+1;
        end;
        for i=frameNo(1):frameNo(end)
             %%eliminamos blobs fuera de la maskara de procesamiento
            center.x=x+width/2;
            center.y=y+height/2;
            if(center.x>processing_mask.x && center.x<processing_mask.x+processing_mask.w)
                if(center.y>processing_mask.y && center.y<processing_mask.y+processing_mask.h)
                    bbGT{object_number}(i+1, :) = [x, y, width, height];
                else
                    bbGT{object_number}(i+1, :) = [0, 0, 0,0];
                end
            else
                bbGT{object_number}(i+1, :) = [0, 0, 0,0];
            end
            
        end;
    end;
end;
% % eliminablobs blobs que sean todo ceros
Blobs_track={};
count=1;
for i=1:size(bbGT,2)
    index=find(bbGT{i}~=0);
    if(~isempty(index))
        Blobs_track{count}=bbGT{i};
        count=count+1;
    end
end

%transformamos a formato de salida Blobs
Blobs={};


for i=1:size(Blobs_track,2)
    for j=1:size(Blobs_track{i},1)
        X=Blobs_track{i}(j,1);
        Y=Blobs_track{i}(j,2);
        W=Blobs_track{i}(j,3);
        H=Blobs_track{i}(j,4);
        if(X~=0 && Y~=0 && W~=0 && H~=0)
            if X<0
                blob.x=0;
            else
                blob.x=X;
            end
            if Y<0
                blob.y=0;
            else
                blob.y=Y;
            end
            blob.h=H;
            blob.w=W;
            blob.num_frame=j;
            blob.objectid=i;
            blob.score=1;
            if size(Blobs,2)>=j
                aux=Blobs{j};
            else
                aux=[];
            end
            %%eliminamos blobs fuera de la maskara de procesamiento
            center.x=blob.x+blob.w/2;
            center.y=blob.y+blob.h/2;
            if(center.x>processing_mask.x && center.x<processing_mask.x+processing_mask.w)
                if(center.y>processing_mask.y && center.y<processing_mask.y+processing_mask.h)
                    Blobs{j}=[aux blob];
                    num_blobs_total=num_blobs_total+1;
                end
            end
            
        end
    end
end

status = fclose(fidIn);
if status == -1
    error('Close file %s failed!', video_name);
end;


