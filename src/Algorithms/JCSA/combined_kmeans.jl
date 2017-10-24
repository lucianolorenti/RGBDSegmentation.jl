using ScikitLearn
using Distances
@sk_import cluster: KMeans
"""
```
function initial_centroids(plain_data,  k::Integer)
```
"""
function initial_centroids(plain_data,   k::Integer)

    #Three centroids(RGB, XYZ, N) per cluster
    centroids = zeros(3,3,k)
    for i=1:3
        local model =  ScikitLearn.fit!(KMeans(init="k-means++", n_clusters=k, n_init=2, max_iter=100, ),plain_data[i]')
        local labels = model[:labels_]+1
        for j=1:k
            centroids[:,i,j] = mean(plain_data[i][:,find(labels.==j)],2)
			centroids[:,i,j] /= norm(centroids[:,i,j])
        end
    end
    return centroids
end
"""
```
function plain_data(RGB::Array{<:Number,3},
                          XYZ::Array{T1,3},
                          N::Array{T2,3}) where T1<:AbstractFloat where T2<:AbstractFloat

```
Converts every image 'RGB', 'XYZ', and 'N' wich have the following size `size(I)=(n_rows,n_cols,n)` to an array of size `size(A)=(n, n_rows*n_cols)`
"""
function plain_data(RGB::Array{<:Number,3},
                          XYZ::Array{<:AbstractFloat,3},
                          N::Array{<:AbstractFloat,3})
    (nr,nc,nd)   = size(XYZ)
    local p_data   = [zeros(3,nr*nc),zeros(3,nr*nc),   zeros(3,nr*nc)]
    local data         = [RGB, XYZ, N]
    for j=1:3
        local  idx   = 1
        for p in CartesianRange((nr,nc))
            p_data[j][:,idx ] = data[j][p[1],p[2],:]
            idx+=1
        end
    end
    return p_data
end
function indmin(x)
    (vals, idx) = findmin(x,2)
    idx = ind2sub(size(x),vec(idx))[2]
    return (vals,idx)
end
"""
```
function initial_clusters(RGB::Array{<:Number,3},
                          XYZ::Array{T1,3},
                          N::Array{T2,3},
                          k::Integer) where T1<:AbstractFloat where T2<:AbstractFloat
```
Return the initial assigments using a combined distance among the patterns of the three images.
"""
function initial_clusters(RGB::Array{<:Number,3},
                          XYZ::Array{<:AbstractFloat,3},
                          N::Array{<:AbstractFloat,3},
                          k::Integer)

    (nr,nc,nd)   = size(XYZ)
    local p_data     = plain_data(RGB,XYZ,N)
    local centroids  = initial_centroids(p_data, k)
    local diff       = 10000;
    local epsilon    = 0.1;
    local value      = 100;
    local iteration  = 1;
    local maxnumIter = 50;
	local clust      = nothing;
    while (diff > epsilon) && (maxnumIter>iteration)
        iteration = iteration+1;
        oldvalue      = value;
        distance_to_center = zeros(nr*nc, k)
        distances = [Euclidean(), Euclidean(), CosineDist()]
        for i=1:3
            for j=1:k
                distance_to_center[:,j] += colwise(distances[i],p_data[i], vec(centroids[:,i,j]))
            end
        end
        # assign points to nearest cluster (based on minimum distance)
        (distMin,clust) = indmin(distance_to_center)
        value = sum(distMin);
        # compute new cluster centroids
	    for h=1:k
    	    try
				local cluster_indices = clust.==h
				for j=1:2
            		centroids[:,j,h] = mean(p_data[j][:,cluster_indices],2)
           		end
				centroids[:,3,h] = sum(p_data[3][:,cluster_indices])
            	centroids[:,3,h] = centroids[:,3,h]./ norm(centroids[:,3,h]);
	        catch
    	        succ = 0;
         	    return;
	        end
		end

		indices_invalid_centroid = vec(find(isnan.(centroids)))
    	if (!isempty(indices_invalid_centroid))
	        tindices = ind2sub(size(centroids),indices_invalid_centroid)[3]
        	centroids = centroids[:,:,setdiff(1:k, tindices)]
		    k = k-length(tindices);
		end
		diff = abs(value - oldvalue);
	end
    return (clust, p_data)
end
