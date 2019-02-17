using Distances
using Clustering
using Statistics
using LinearAlgebra
using Distances
using RGBDSegmentation: CDNImage
"""
```
function initial_centroids(plain_data,  k::Integer)
```
Return a ``9 \times k`` matrix containing the centroids of each  cluster.

Each centroid is represented as a 9-dimensional vector in which the 
first three components represent the color components,
the 4:6 indices represent the spatial components and 
the last three component represent the normal component.

Each slice of the array are normalized. i.e. 
``norm(c[1:3, j]) = 1``

"""
function initial_centroids(plain_data,   k::Integer)
    centroids = zeros(9, k)
    indices = [1:3, 4:6, 7:9]
    for i=1:3
        model = kmeans(
            plain_data[indices[i], :],
            k,
            maxiter=500)
        labels = assignments(model)
        for j=1:k
            centroids[indices[i], j] = mean(
                plain_data[indices[i], findall(labels.==j)],
                dims=2)
	    centroids[indices[i], j] /= norm(centroids[indices[i], j])
        end
    end
    return centroids
end
function indmin(x)
    (vals, cidxs) = findmin(x, dims=2)
    idx = [cidx[2] for cidx in cidxs]
    return (vals,idx)
end
"""
```
function initial_clusters(p_data::Array{T, 2},
                          k::Integer) where T <: Number
```
Return the initial assigments using a combined distance among the patterns of the three images.
"""
function initial_clusters(p_data::Array{T, 2},
                          k::Integer) where T <: Number

    centroids = initial_centroids(p_data, k)
    indices = [1:3, 4:6, 7:9]
    diff = 10000;
    epsilon = 0.1;
    value = 100;
    iteration = 1;
    maxnumIter = 50;
    clust = nothing;
    while (diff > epsilon) && (maxnumIter>iteration)
        iteration = iteration+1;
        oldvalue = value;
        distance_to_center = zeros(size(p_data,2), k)
        distances = [Euclidean(), Euclidean(), CosineDist()]
        for i=1:3, j=1:k
            distance_to_center[:,j] += colwise(
                distances[i],
                p_data[indices[i],:],
                vec(centroids[indices[i],j])
            )
        end
        # assign points to nearest cluster (based on minimum distance)
        (distMin,clust) = indmin(distance_to_center)
        value = sum(distMin);
        # compute new cluster centroids
	for h=1:k
    	  try
	    cluster_indices = vec(clust.==h)
            centroids[1:3, h] = mean(
                p_data[1:3, cluster_indices],
                dims=2)
            centroids[4:6, h] = mean(p_data[4:6,cluster_indices],
                                    dims=2)
            centroids[7:9, h] = sum(p_data[7:9,cluster_indices],
                                    dims=2)
            centroids[7:9, h] = centroids[7:9, h]./ norm(centroids[7:9, h]);
	   catch e
                
                succ = 0;
         	return;
	   end
	end
	indices_invalid_centroid = unique([id[2] for id in vec(findall(isnan.(centroids)))])
    	if (!isempty(indices_invalid_centroid))
            centroids = centroids[:, setdiff(1:k, indices_invalid_centroid)]
	    k = k - length(indices_invalid_centroid);
	end
	diff = abs(value - oldvalue);
    end
    return clust
end
