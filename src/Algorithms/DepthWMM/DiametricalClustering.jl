module DiametricalClustering
using StatsBase
function initialAssignment(M::Matrix, k::Integer)
    (m,n) = size(M)
    local indexes =  sample(1:k,m)
    return indexes
end
function clusteringError(G::Matrix{Float32},vectors::Vector{Vector{Float32}}, assignments, k::Integer)
    local cerror = 0
    for j=1:k
        local cluster = find(assignments.==j)
        for c in cluster
            cerror = cerror + dot(G[c,:], vectors[j])^2
        end
    end
    return  cerror
end

function initialCentroids(M::Matrix{<:AbstractFloat}, k::Integer)
    (d,n) =size(M)
    Mu = zeros(d,k)
    vectorMean = vec(mean(M,2))
    Mu0 = vectorMean/norm(vectorMean)
    perturbation = 0.1
    for i=1:k
        randomVector = rand(d)-0.5
        randomNorm   = perturbation*rand()
        smallRandVec =  randomNorm*randomVector/norm(randomVector)
        v            =  Mu0 + smallRandVec;
        Mu[:,i]      =  v/norm(v)
    end
    return Mu
end
function clusterize(X::Matrix{<:AbstractFloat}, k::Integer; maxIterations::Integer = 5000, threshold = 0.00000001)
    local mu = initialCentroids(X,k)
    local assignments = nothing
    local diff    = 1;
    local epsilon = 0.001;
    local value   = 100;
    local iteration = 1;
    local oldvalue = 5
    while (value-oldvalue > epsilon && iteration<200)
        iteration = iteration+1;

        oldvalue  = value;
        # assign points to nearest cluster
        simMat          =  (mu'*X).^2;
        (simax,indexes) =  findmax(simMat,1);

        assignments   = ind2sub(size(simMat),vec(indexes))[1]

        #compute objective function value
        value         = sum(simax);
        # compute cluster centroids
        for h=1:k
            if length(find(assignments.==h)) > 0
                clustData = X[:,find(assignments.==h)];
                (svd,_)   = svds(clustData',nsv=1);
                V = svd[:V]
                mu[:,h]   = V';
            end
        end
    end
    return (assignments, mu)
end
end
