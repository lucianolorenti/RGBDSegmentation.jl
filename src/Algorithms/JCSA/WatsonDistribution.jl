using StatsBase

"""
https://stats.stackexchange.com/questions/111759/exponential-family-observed-vs-expected-sufficient-statistics
"""


mutable struct MvWatson
    u::Vector{Float64}
    k::Float64
end
const SufficientStatistic = Vector

mutable struct MvWatsonExponential <: ExponentialFamily
    dim::Integer
    concentration::Float64
    n::Vector{Float64}
    theta::Vector{Float64}
    DLNF::Float64
end

function MvWatsonExponential(n::Vector{Float64})
    dim = round(Int64,(-1+sqrt(1+(8*length(n))))/2)
    w   = MvWatsonExponential(dim, 1, n, Float64[],0)
    update_parameters!(w)
    return w
end
"""
Apply Newton-Raphson method root finder  to approximate ``κ`` from
 ``\\vert \\vert n \\vert \\vert_2`` using the following iterative update iteration:
 ``k_{l+1} = k_l - \\dfrac{q(a,c;k_l) - \\vert \\vert n \\vert \\vert_2}{q'(a,c,k_l)} ``

"""
function estimate_concentration(dim::Integer, n::Vector{<:AbstractFloat}; tol::Float64 = 0.000001)
    a     = 0.5
    c     = (dim)/2
    r = norm(n)
    kappa_upper_bound = 500;
    k = 1.0
    diff = 100;
    ii=2;
    prevk = Inf
    while  abs(k - prevk) > tol && (k < kappa_upper_bound)
        prevk=k
        q = kummer_ratio(a,c,k, tol = eps()*2)
        dq = (1 - (c/k))*q + (a/k) -q^2
        k = k - ( (q - r) / (dq))
    end
    return k
end
function update_parameters!(watson::MvWatsonExponential)
    d = dimension(watson)
    watson.concentration = estimate_concentration(d,watson.n);
    g_norm_theta = kummer_ratio(0.5,d/2, watson.concentration)
    R_norm_theta = g_norm_theta/ watson.concentration;
    watson.theta = watson.n ./ R_norm_theta; # natural parameter
    LNF = log(M(0.5, d/2, watson.concentration));# The log normalizing function
    watson.DLNF = dot(watson.theta, watson.n) - LNF; # Dual (expectation parameter) of the log normalizing function

end
function bregman_divergence(wmm::MvWatsonExponential, x::Vector)
    n     = wmm.n
    theta = wmm.theta
    return  wmm.DLNF +  dot(x-n,theta)
end
"""
Computes the bregman divergence of every pattern located in the columns of x
"""
function bregman_divergence(wmm::MvWatsonExponential, x::Matrix)
    n = wmm.n
    theta = wmm.theta
    res =  wmm.DLNF .+ theta'*(x.-n)
    return res
end
function bregman_divergence(wmm1::MvWatsonExponential, wmm2::MvWatsonExponential)
    n1 = wmm1.n
    n2 = wmm2.n
    theta2 = wmm2.theta
    return  wmm1.DLNF - wmm2.DLNF - dot(n1-n2,theta2)
end

function bregman_divergence(wmm1::MvWatson, wmm2::MvWatson)
    
    a     = 0.5
    c     = dimension(wmm1)/2
    theta1     = wmm1.k* sufficient_statistic(MvWatsonExponential, wmm1.u)
    theta2     = wmm2.k* sufficient_statistic(MvWatsonExponential, wmm2.u)
    return  log(M(a,c,wmm1.k)) - log(M(a,c,wmm2.k)) - dot(theta1-theta2, kummer_ratio(a,c,wmm2.k)*(theta2/wmm2.k))
end
function sufficient_statistic_dimension(d::Integer)
    return round(Integer,d+ (d*(d-1))/2)
end
function MvWatsonExponential(dim::Integer)
    p = sufficient_statistic_dimension(dim)
    return MvWatsonExponential(dim,0.0, zeros(p), zeros(p), 0.0)
end
function dimension(w::MvWatsonExponential)
    return w.dim
end
function sufficient_statistic(::Type{MvWatsonExponential}, d::Integer, n::Integer=1)
    p = sufficient_statistic_dimension(d)
    if n==1
	return zeros(p)
    else
	return zeros(p,n)
    end
end
function sufficient_statistic(t::Type{MvWatsonExponential}, x::Vector{<:AbstractFloat})
    y = sufficient_statistic(t,length(x))
    sufficient_statistic!(t,y,x)
    return y
end
function sufficient_statistic(t::Type{MvWatsonExponential}, x::AbstractArray{<:Number,2})
    y = zeros(
        sufficient_statistic_dimension(size(x,1)),
        size(x,2))
    for i=1:size(x,2)
        y[:, i] = sufficient_statistic(t,x[:, i])
    end
    return y
end

function sufficient_statistic(t::Type{MvWatsonExponential},  x::Array{<:AbstractFloat,3})
    y = zeros(
        sufficient_statistic_dimension(size(x,1)),
        size(x,2)*size(x,3))
    idx = 1
    for i=1:size(x,2), j=1:size(x,3)
        y[:,idx] = sufficient_statistic(t,x[:, i ,j])
        idx += 1
    end
    return y
end
function sufficient_statistic!(::Type{MvWatsonExponential}, y::Vector{<:AbstractFloat}, x::Vector{<:AbstractFloat})
    y[1:length(x)] .= x.^2
    r = length(x)+1
    for j=1:length(x)-1
        for i=j+1:length(x)
            y[r] = x[j]*x[i]*sqrt(2)
            r=r+1
        end
    end
end
function dimension(d::MvWatson)
    return length(d.u)
end
"""

Kummer’s confluent hypergeometric function
https://www.mathworks.com/matlabcentral/fileexchange/29766-confluent-hypergeometric-function?focused=5192808&tab=function

 Estimates the value by summing powers of the generalized hypergeometric
 series:

 ``sum(n=0-->Inf)[(a)_n*x^n/{(b)_n*n!}``

 until the specified tolerance is acheived.

"""
function M1(a::Float64, b::Float64, z::T; tol=0.00001) where T<:AbstractFloat
    term = BigFloat((z*a)/b)
    f = BigFloat(1 + term)
    n = 1;
    an = a
    bn = b;
    nmin = 50;
    while (n < nmin) || abs(term) > tol
        n = n + 1;
        an = an + 1;
        bn = bn + 1;
        term = (term*an*z)/(bn*n);
        prevf = f
        f = f + term;
        if isinf(f)
            return Inf
        end
    end
    return Float64(f)
end
function kummer_ratio(a::Float64, c::Float64, x::T; tol::Float64=0.00001) where T<:AbstractFloat
    val =  (a/c)*(M(a+1,c+1,x)/(M(a,c,x)))
    if isnan(val)
        return 1
    else
        return val
    end
end
import Distributions.pdf
import Distributions.logpdf!
function pdf(w::MvWatson,x::Vector{<:AbstractFloat})::Float64
    p = dimension(x)
    return ((gamma(p/2)/(2*pi^(p/2)*M(0.5,p)/2,w.k)))*exp(w.k*(dot(w.u,x)^2))
end
function pdf(w::MvWatson, m::Matrix{<:AbstractFloat})
    p = dimension(w)
    c = ((gamma(p/2)/(2*pi^(p/2)*M(0.5,p)/2,w.k)))
    return prod(c*exp.(w.k.*(w.u'*v).^2))
end
function logpdf!(output, w::MvWatson, data::Matrix)
    dim = size(data,2)
    coeff = log(1/M(0.5,dim/2,w.k))
    output .=vec(w.k.*(data'*w.u).^2 .+ coeff)
    ind = output.>700;
    output[ind] = output[ind] .- 1000;
end

function M(a,b,x)
    nl = 0
    ta=[]
    tb=[]
    xg=[]
    tba=[]
    a0=a;
    a1=a;
    x0=x
    hg=0.0
    if (b == 0.0 || b == -abs(floor(b)))
	hg=1.0+300;
    elseif (a == 0.0 || x == 0.0)
	hg=1.0;
    elseif (a == -1.0)
    	hg=1.0-x./b
    elseif (a == b)
    	hg=exp(x)
    elseif (a-b == 1.0)
    	hg=(1.0+x./b).*exp(x)
    elseif (a == 1.0&&b == 2.0)
    	hg=(exp(x)-1.0)./x
    elseif (a == floor(a)&& a < 0.0)
    	m=floor(-a)
	r=1.0
    	hg=1.0
	for  k=1:m
	    r=r.*(a+k-1.0)./k./(b+k-1.0).*x
	    hg=hg+r
	end
    end
    if (hg != 0.0)
	return hg
    end
    if (x < 0.0)
	a=b-a
	a0=a
    	x=abs(x)
    end
    if (a < 2.0)
	nl=0
    end
    if (a >= 2.0)
	nl=1
	la=floor(a)
	a=a-la-1.0
    end
    y0 = 0
    y1 = 1
    
    for n=0:nl
	if (a0 >= 2.0)
	    a=a+1.0
	end
	if (x <= 30.0+abs(b)||a < 0.0)
    	    hg=1.0
            rg=1.0
	    for  j=1:500
    	        rg=rg.*(a+j-1.0)./(j.*(b+j-1.0)).*x
        	hg=hg+rg;
            	if (abs(rg./hg) < 1.0e-15)
		    break
		end
    	    end
	else
    	    (a,ta)=gamma(a,ta)
	    (b,tb)=gamma(b,tb)
    	    xg=b-a
	    (xg,tba)=gamma(xg,tba)
	    sum1=1.0
	    sum2=1.0
	    r1=1.0
	    r2=1.0
	    for  i=1:8
	        r1=-r1.*(a+i-1.0).*(a-b+i)./(x.*i)
	        r2=-r2.*(b-a+i-1.0).*(a-i)./(x.*i)
	        sum1=sum1+r1
	        sum2=sum2+r2
	    end
	    hg1=tb./tba.*x.^(-a).*cos(pi.*a).*sum1;
	    hg2=tb./ta.*exp(x).*x.^(a-b).*sum2;
    	    hg=hg1+hg2;
	end
    	if (n == 0)
	    y0=hg
	end
    	if (n == 1)
	    y1=hg
	end
    end
    if (a0 >= 2.0)
	for  i=1:la-1
	    hg=((2.0.*a-b+x).*y1+(b-a).*y0)./a;
    	    y0=y1;
            y1=hg;
	    a=a+1.0;
    	end
    end
    if (x0 < 0.0)
	hg=hg.*exp(x0)
    end
    a=a1
    x=x0
    if isinf(hg)
        return realmax()
    else
        return hg
    end
end

function gamma(x,ga)
    g=zeros(26);
    if (x == floor(x))
    	if (x > 0.0)
            ga=1.0
	    m1=x-1
    	    for  k=2:m1
        	ga=ga.*k;
	    end
    	else
            ga=1.0e300
	end
    else
    	if (abs(x) > 1.0)
	    z=abs(x)
    	    m=floor(z);
	    r=1.0;
	    for  k=1:m;
    	        r=r.*(z-k);
	    end;
    	    z=z-m;
	else
    	    z=x;
	end
	g=[1.0,0.5772156649015329,-0.6558780715202538,-0.420026350340952e-1,0.1665386113822915,-.421977345555443e-1,-.96219715278770e-2,.72189432466630e-2,-.11651675918591e-2,-.2152416741149e-3,.1280502823882e-3,-.201348547807e-4,-.12504934821e-5,.11330272320e-5,-.2056338417e-6,.61160950e-8,.50020075e-8,-.11812746e-8,.1043427e-9,.77823e-11,-.36968e-11,.51e-12,-.206e-13,-.54e-14,.14e-14,.1e-15];
        gr=g[26]
	for  k=25:-1:1
            gr=gr.*z+g[k];
	end
    	ga=1.0./(gr.*z)
	if (abs(x) > 1.0)
    	    ga=ga.*r;
            if (x < 0.0)
		ga=-pi./(x.*ga.*sin(pi.*x))
	    end
    	end
    end
    return (x,ga)
end
"""
Return the same dsitribution using the natural parameters
"""
function  source_parameters(m::MvWatsonExponential)
    μ = source_vector(MvWatsonExponential, m.n)
    return MvWatson(μ/norm(μ), m.concentration)
end
function source_vector(m::Type{MvWatsonExponential}, W::SufficientStatistic)
    dim = round(Int64,(-1+sqrt(1+(8*length(W))))/2)
    x_init = sqrt.(W[1:dim])
    W = W.>0
    x = x_init
    if (W[4] & !W[5] & !W[6])
        x[3] = -x_init[3]
    elseif !W[4] && !W[5] & W[6]
        x[1] = -x_init[1]
    elseif !W[4] && W[5] && !W[6]
        x[2] = -x_init[2]
    end
    return x
end

# The Multivariate Watson Distribution: Maximum-Likelihood Estimation and other Aspects
#Suvrit Sra and Dmitrii Karp
import Distributions.fit_mle
function L(r::T, c::T, a::T) where T<:Number
    return ((r*c-a)/(r*(1-r)))*(1+((1-r)/(c-a)))
end
function B(r::T, c::T, a::T) where T<:Number
    return ((r*c-a)/(2*r*(1-r)))*(1+sqrt(1+ ((r*(c+1)*r*(1-r))/(a*(c-a)))))
end
function U(r::T, c::T, a::T) where T<:Number
    return ((r*c-a)/(r*(1-r)))*(1+(r/a))
end
function concentration(r::T1, c::T2, a::T3) where T1<:Number where T2<:Number where T3<:Number
    if (r>0) && (r<a/(2*c))
        return U(r,c,a)
    elseif (r>=(a/(2*c))) && (r<((2*a)/(2*sqrt(c))))
        return B(r,c,a)
    elseif (r>=((2*a)/(2*sqrt(c)))) && (r<1)
        return L(r,c,a)
    end
end
function concentration(v::Vector, S::Matrix, c::Float64, a::Float64)
    r = v'*S*v
    if (r<=0)
        r=eps()
    end
    if r>=1
        r=1-5*eps()
    end
    return concentration(float64(r), float64(c), float64(a))
    
end

import Distributions.loglikelihood
function  loglikelihood(d::MvWatson, X::AbstractMatrix)
    n = size(X,2)
    p = size(X,1)
    S = StatsBase.scattermat_zm(X,2)/n
    return loglikelihood_from_scatter(d, n, S)
end
function loglikelihood_from_scatter(d::MvWatson, n::Integer, S::AbstractMatrix)
    r = d.u'*S*d.u
    p = size(S,1)
    return n*(d.k*r - log(M(1/2, p/2, d.k)))
end
function fit_mle(m::Type{MvWatson}, X::Matrix)
    n = size(X,2)
    p = size(X,1)
    S = StatsBase.scattermat_zm(X,2)/n
    a = 1/2
    c = p/2
    
    (eval,evec) = eig(S)
    w_1 = MvWatson(evec[:,1], concentration(evec[:,1], S, c, a))
    w_2 = MvWatson(evec[:,3], concentration(evec[:,3], S, c, a))
    
    llh1 = loglikelihood_from_scatter(w_1, n, S)
    llh2 = loglikelihood_from_scatter(w_2, n, S)
    
    if (llh1 > llh2)
        return w_1
    else
        return w_2
    end   
    
end
#=facts("Watson Distribution") do
we = MvWatsonExponential(3)
@fact sufficientStatistic( [3.0;1.0;5.0]) --> roughly([9.0; 1.0; 25.0; 4.24264; 21.2132; 7.07107]; atol=0.0001)
@fact M(0.5, 3/2, 0.5) --> roughly(1.19496, atol=0.001)
@fact kummer_ratio(0.5, 3/2, 0.5) --> roughly(0.37973, atol=0.001)
end=#
