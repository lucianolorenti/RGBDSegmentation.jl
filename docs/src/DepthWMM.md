
# Unsupervised Clustering of Depth Images using Watson Mixture Model
* [Tesis Doctoral](https://tel.archives-ouvertes.fr/tel-01160770v2/document)
* [Articulo](http://ieeexplore.ieee.org/document/6976757/)
* [Dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2)
## Familia de exponenciales
Una función de densidad de probabilidad $f(\mathbf{x} | \theta) $ pertenece a la familia de exponenciales si tiene la siguiente forma:
$$f(\mathbf{x}|\theta) = \exp( (t(\mathbf{x}) \cdot  \theta )) -F(\theta) + k(\mathbf{x}) )$$
Aquí, $t(\mathbf{x})$ es el estadístico suficiente, $\theta$ es el parámetro natural, $F(\theta)$ es la función de normalización logarítmica, $k(\mathbf{x})$ es la carrier measure. La esperanza del estadístico suficiente $E[t(\mathbf{x})]$ es llamada el parámetro de esperanza $(n)$. Existe una correspondencia uno a uno entre $n$ y $\theta$:
$$n=\nabla_\theta F(\theta) \mbox{  y  } \theta = (\nabla_\theta F(\theta))^{-1}(n)$$
## Divergencia de Bregman

Las divergencias de Bregman generalizan un número de function de distorsión que son usadas comunmente en clusterin. Una distribución de probabilidad puede tomar bedeficion de la divergencia de Bregman si se puede representar en su forma de familia de exponencial canónica.
La divergencia de Bregman con un parámetro de esperanza $n$ puede ser definido como:
$$D_G(n_1, n_2) = G(n_1) - G(n_2) - ((n_1 -n_2) \cdot \nabla G(n_2))$$
Donde $G(.)$ es el Legendre dual de $F(.)$.
$$\begin{align*}G(n) &= ((\nabla F)^{-1}(n) \cdot n)-F((\nabla F)^{-1}(n))\\
  &= \theta \cdot n - F(\theta)\end{align*}$$
Entonces

$$D_G(n_1, n_2) = \theta_1 \cdot n_1 - F(\theta_1) - \theta_2 \cdot n_2 + F(\theta_2) - ((n_1 -n_2) \cdot \nabla G(n_2))$$
## Distribución Watson
La distribución de Watson multivariada (mWD) es un distribución fundamental que modela datos direccionales simétricamente axiales. Para un vector unitario d-dimensional simétricamente axiales $\pm \mathbf{x} = [x_1, ...,x_d]^T \in S^{d-1} \subset R^d$, la distribución multivariada de Watson se define como:
$$W_d(\mathbf{x}|u,k) = M(a,c,k)^{-1} \exp(k(u^t \mathbf{x})^2)$$. Aquí u es la dirección media, k es la concentración, $a=\frac{1}{2}$, $c={d}{2}$ y $M(a,c,k)$ es la función hipergeométrica confluente de Kummer.

Con el objetivo de obtener una representación de familia de exponenciales canónica para una distribución de watson podemos reescribirla como:
$$W_d(\mathbf{y}|v,k) = \exp(kv^t \mathbf{y} - log M(k))$$
con $\mathbf{y},v \in R^p, p=d+C^d_2$:
$$\begin{align*}_
\mathbf{y} &= [ x_1^2,...,x_d^2, \sqrt{2}x_1x_2,..., \sqrt{2}x_{d-1}x_d ]^T \\
v          &= [u_1^2,...,u_d^2, \sqrt{2}u_1 u_2,...,\sqrt{2}u_{d-1} u_d]^T
\end{align*}$$
Podemos descomponer la distribución de la siguiente forma:
* El estadístico suficiente $t(\mathbf{x})  = \mathbf{y}$
* El parámetro natural $\theta = kv$
* La función de normalización logarítmica $F(0) = log M(k)$
* Carrier Measure $k(\mathbf{x}) = 0$

Luego, podemos escribir $v$ y $k$ en términos de los parámetros naturales $\theta$ como:
* $\theta = kv$
* $v = \dfrac{\theta}{\vert \vert \theta \vert \vert_2}$
* $k = \vert \vert \theta \vert \vert_2$

Ahora, podemos escribir el gradiente de la función $F(\theta)$ como:
$$n=\nabla_\theta F(\theta) = q(a,c,k)\frac{\theta}{k}$$

donde $q(a,c,k)$ es llamado la proporción Kummer:
$$q(a,c,k) = \frac{M'(k)}{M(k)} = \frac{a}{c} \frac{M(a+1,c+1,k)}{M(a,c,k)}$$
Considerando que $\theta = \nabla_n G(n)$

$$\begin{align*}
D_G(n_1, n_2) &= \theta_1 \cdot n_1 - F(\theta_1) - \theta_2 \cdot n_2 + F(\theta_2) - ((n_1 -n_2) \cdot \nabla G(n_2)) \\
 &= \theta_1 \cdot n_1 - \log M(k_1) - \theta_2 \cdot n_2 + \log M(k_2) - ((n_1 -n_2) \cdot \theta_2) \\
 &= \theta_1 \cdot n_1 - \log M(k_1) - \theta_2 \cdot n_2 + \log M(k_2) - n_1 \cdot \theta_2 + n_2 \cdot \theta_2 \\
 &= \theta_1 \cdot n_1 - \log M(k_1) + \log M(k_2) - n_1 \cdot \theta_2  \\
& \mbox{SI lo parámetros de concentración son iguales} \\
 &= \frac{n_1 k}{q(a,c,k)} \cdot n_1 -  n_1 \cdot \frac{n_2 k}{q(a,c,k)}  \\

 &= \frac{n_1 k \cdot n_1 - n_1 \cdot n_2 k}{q(a,c,k)}  \\

 &= \frac{k - n_1 \cdot n_2 k}{q(a,c,k)}  \\

 &= \frac{k( 1 -  n_1 \cdot n_2)}{q(a,c,k)}  \\

\end{align*}
$$
## Obtención de $k$ a partir de n

Se aplica el metodo de Newton-Raphson para aproximar el valor de $k$ a partir de  $\vert \vert n \vert \vert_2$ utilizando el esquema
iterativo que se detalla a continuación:
$$k_{l+1} = k_l - \dfrac{q(a,c;k_l) - \vert \vert n \vert \vert_2}{q'(a,c,k_l)}$$
donde $q'(a,c;k)$ es la primer derivada de la proporción Kummer y puede ser calculada como:
$$q'(a,c;k) = (1-\frac{c}{k})q(a,c;k) +  \frac{a}{k}-q(a,c;k)^2$$

## Primer paso
CLusterinzar con diametrical clustering.
1. $n_i = \sum\limits_{j \in C_i} \dfrac{t(x_i)}{|C_i|}$
2. $\pi_i = \dfrac{|C_i|}{N}$
## Paso de expectación

* Tenemos que $\theta = \nabla_n G(n) = \dfrac{nk}{q(a,c;k)}$ . Se puede precalcular para cada paso
$$\begin{align*}.
p(\gamma_i = j | \mathbf{x}_i)  &= \frac{\pi_{j} \exp (G(n_{j}) + (t(\mathbf{x_i}) - n_{j}) \cdot \nabla G(n_j))}{\sum\limits_{l} \pi_{l} \exp (G(n_{l}) + (t(\mathbf{x_i}) - n_{l}) \cdot \nabla G(n_l))}\\
&= \dfrac{\pi_{j} \exp (\theta_j \cdot  n_{j} - \log M(k_{j}) + (y_i - n_{j}) \cdot \theta_j)}{\sum\limits_{l} \pi_{l} \exp (G(n_{l}) + (t(\mathbf{x_i}) - n_{l}) \cdot \nabla G(n_l))}
\end{align*}$$
## Paso de maximización

## Asignación de los patrones
## Ejemplo de uso
```julia
using FITSIO
using TOFSegmentationBenchmark
using SegmentationBenchmark
using RGBDProcessing
using Iterators
using  PyPlot

function swapDimensions(img)
    (ff, nr,nc) = size(img)
    local N1 = zeros(nr,nc,3)
    for (r,c) in product(1:nr,1:nc)
	    N1[r,c,:] = img[:,r,c]
     end
    for j=1:3
      N1[:,:,j] = (N1[:,:,j] + minimum(N1[:,:,j])) /( maximum(N1[:,:,j]) - minimum(N1[:,:,j]))
    end
    return N1

end
#path            = "/home/luciano/ownCloud/Capturas/SR4000/Etiquetadas"
#files           = [joinpath(path,img_path) for img_path in readdir(path)]
normal_params = Dict{Any,Any}("rectSize"=> 2.0,
                              "normalSmoothingSize"=>2.0, "maxDepthChangeFactor"=>2.0)
f = FITS("/home/luciano/datasets/image_nyu_v2_211.fits", "r+")
N = nothing
if (length(f)==3)
    R = read(f[2])
    points3d = RGBDProcessing.rgb_plane2rgb_world(R)
    (N1, N, N2) = RGBDProcessing.compute_local_planes(points3d,window=5,relDepthThresh=0.05)
    write(f,N)
end
N = read(f[4])
close(f)
using Iterators
using DepthWMM
nr=size(N,2)
nc=size(N,3)
normals = reshape(N,3,size(N,2)*size(N,3))
imgs=nothing
b  = clusterize(normals,5)
using PyPlot
imshow(reshape(b,nr,nc))
```
```
imshow(swapDimensions(N))
N        = RGBDProcessing.get_normal_image(points3d,PCL.Average_3D_Gradient, rectSize=7.0, normalSmoothingSize=3.0, maxDepthChangeFactor=0.01)
imshow(swapDimensions(N ))
```
