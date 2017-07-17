function Blobs_out=Blobs_Threshold(Blobs,threshold);

for i=1:size(Blobs,2)
    Blobs_out{i}=[];
    for j=1:size(Blobs{i},2)
        if Blobs{i}(j).score>=threshold
            Blobs_out{i}=[Blobs_out{i},Blobs{i}(j)];
        end
    end
end