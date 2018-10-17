"""
Unsupervised Clustetaring of Depth Images using Watson Mixture Mode
Tesis Doctoral: https://tel.archives-ouvertes.fr/tel-01160770v2/document
[Unsupervised Clustering of Depth Images Using Watson Mixture Model](http://ieeexplore.ieee.org/document/6976757/)
"""
module DepthWMM
export WatsonMixtureModel,
      clusterize

include("DiametricalClustering.jl")
include("../JCSA/ExponentialDistributions.jl")
#using Distributions
using Clustering

mutable struct WatsonMixtureModel
    dim::Integer
    weights::Vector{Float64}
    watsons::Vector{MvWatsonExponential}
end

function WatsonMixtureModel(dim::Integer, k::Integer)
    weights = zeros(k)
    watsons = MvWatsonExponential[]
    for j=1:k
        push!(watsons, MvWatsonExponential(dim))
    end
    return WatsonMixtureModel(dim, weights, watsons)
end
function dimension(w::WatsonMixtureModel)
    return w.dim
end
function calculateV( wd::MvWatsonExponential, dim::Integer, data::Vector{<:Vector{<:AbstractFloat}})
  n  = length(data)
  x  = sufficientStatistic(wd,dim)
  for i=1:n
      x = x + sufficientStatistic(data[i])
  end
  return x/n
end
function expectation(wmm::WatsonMixtureModel, x::Matrix{<:AbstractFloat})
    local k = length(wmm.weights)
    (dim, npatterns) = size(x)
    local pij = zeros(Float64, k,npatterns)
    local d = dimension(wmm)
    local a=0.5
    for i=1:npatterns
        xx = x[:,i]
        for j=1:k
            pij[j,i] = wmm.weights[j]*exp(bregmanDivergence(wmm.watsons[j],xx))
        end
        pij[:,i] = pij[:,i]/sum(pij[:,i])
    end
    return pij
end
function maximization(wmm::WatsonMixtureModel,pij::Matrix{<:AbstractFloat}, x::Matrix{<:AbstractFloat})
    k = length(wmm.weights)
    (dim,nvectors) = size(x)
    for j=1:k
        fill!(wmm.watsons[j].n,0)
        sumpij = 0
        for  i =1:nvectors
            wmm.watsons[j].n = wmm.watsons[j].n + pij[j,i]* x[:,i]
            sumpij = sumpij + pij[j,i]

        end
        wmm.watsons[j].n = wmm.watsons[j].n / (sumpij+eps())
        wmm.weights[j] = (sumpij) /nvectors
    end
end
function _clusterize(wmm::WatsonMixtureModel, x::Matrix{<:AbstractFloat})
    local k          = length(wmm.weights)
    local clustering = zeros(Integer, size(x,2))
    local d = dimension(wmm)
    for j=1:size(x,2)
        vals =[ -bregmanDivergence(wmm.watsons[i], x[:,j]) for i=1:k]
        clustering[j] = indmin(vals)
    end

    return clustering
end
function _clusterize( pij::Matrix{<:AbstractFloat}, k::Integer)
    clustering = ind2sub(size(pij),vec(findmax(pij,1)[2]))[1]
    return clustering
end
function loss(wmm::WatsonMixtureModel, indexes::Vector, x::Matrix{<:AbstractFloat})
    local d = dimension(wmm)
    local k = length(wmm.weights)
    (p,nvectors) = size(x)
    lossSum = 0
    a  = 0.5
    for i=1:length(indexes)
        xx = x[:,i]
        for j=1:k
            lossSum = lossSum + bregmanDivergence(wmm, xx, j)
        end
    end
    return log(-lossSum)
end
function computeParameters(wmm::WatsonMixtureModel)
	local k = length(wmm.weights)
	local d = dimension(wmm)
	for i=1:k
        updateParameters!(wmm.watsons[i])
    end
end
function  getExtensionMatrices(wmm::WatsonMixtureModel, X, mu)
	(d,n) = size(X)
	X_ex = sufficientStatistic(wmm.watsons[1], d, n)
	M_ex = sufficientStatistic(wmm.watsons[1], d, size(mu,2))

	for i=1:n
    	X_ex[:,i] = sufficientStatistic(wmm.watsons[1],X[:,i])
	end
	for i=1:size(mu,2)
		M_ex[:,i] = sufficientStatistic(wmm.watsons[1], mu[:,i])
	end
	return (X_ex, M_ex)
end
function clusterize(x::Matrix{<:AbstractFloat}, k::Integer)
    k1          = k
    k           = k
    wmm         = WatsonMixtureModel(size(x,1),k)
    k           = length(wmm.weights)
    d           = dimension(wmm)

    (assignments, representatives) = DiametricalClustering.clusterize(x,k)
	(xss, mu) = getExtensionMatrices(wmm, x,representatives)
    for i=1:k
        cluster = find(assignments.==i)
        wmm.weights[i] = length(cluster)/length(assignments)
		wmm.watsons[i].n = vec(mean(xss[:,cluster],2) )
	end
	computeParameters(wmm)
    prevLoss = 999999999
    currLoss = 999999998
    #while (currLoss < prevLoss)
    for i=1:300
        pij  = expectation(wmm, xss)
        maximization(wmm,pij,xss )
		computeParameters(wmm)
        println(wmm.weights)
        b = _clusterize(pij,k)

        #currLoss = loss(wmm,b,xss)
        #println(currLoss)
       imshow(reshape(b,480,640))
    end
    MM = rand(k,k)
    M = zeros(k)
    q = zeros(k)
    for i=1:k
        q[i] = kummerRatio(0.5, d/5, wmm.watsons[i].concentration)
        M[i] = log(WatsonDistribution.M(0.5,p/5, wmm.watsons[i].concentration))
    end
    for i=1:k
        ni = wmm.watsons[i].n
        ki = wmm.watsons[i].concentration

        for j=1:k
            kj = wmm.watsons[j].concentration
            nj = wmm.watsons[j].n
            Dij = dot(ni, (ni*ki ) /q[i]) - M[i] +M[j] - dot(ni, (nj*kj)/q[j])
            MM[i,j] = wmm.weights[i] * wmm.weights[j] * Dij
        end
    end
    h = hclust(Symmetric(MM), :average)
    indexes = cutree(h, k=k1)
    println(indexes)

    n = zeros(k1, p)
    pimerged = zeros(k1)
    for i = 1:maximum(indexes)
        pimerged[i] = sum(wmm.weights[indexes.==i])
        for j in find(indexes.==i)
           n[i,:]      =  n[i,:] + wmm.weights[j]*wmm.n[j,:]
        end
        n[i,:] = n[i,:] / pimerged[i]

    end
    wmm.weights = pimerged
    wmm.n      = n
    return _clusterize(wmm,xss)
end

#=facts("DepethWMM") do
    context("Concentrarion") do
	    v  = [0.01;0.05;0.2]
	    k  = norm(v)
        kn = estimateConcentration(length(v),v, tol=0.00000001)
		@fact kummerRatio(0.5, 3/2, kn, tol =0.00000001) --> roughly(k,atol=0.00001)
	end
end=#
end
