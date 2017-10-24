# Unsupervised RGB-D image segmentation using joint clustering and region merging
* [Tesis Doctoral](https://tel.archives-ouvertes.fr/tel-01160770v2/document)
* [Articulo](http://perso.univ-st-etienne.fr/ao29170h/Fichiers/BMVC14UnsupervisedRGBDimagesegmentation.pdf)
* [Dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2)

## Modelo de generación de la imagen
Propose an statistical model that combine the color and shape (3D and surface normals) charactersitics. The model assume that the features are sampled indepentently from a finite mixture of multivariate gausisan (for the color and 3D information) and a Waston distribution (for the surface normals). The model combines $k$ components has the followoing form:
$$ g(x_i | \theta_k) = \sum\limits_{i=1}^k \pi_{j,k}  f_g(x_i^C | u_{j,k}^C, \Sigma_{j,k}^C) f_g(x_i^P | u_{j,k}^P, \Sigma_{j,k}^P) f_w(x_i^N | u_{j,k}^N, k_{j,k}^N)$$
$X_i =\left[ x_i^C, x_i^P, x_i^n \right]$ es el vector característico del píxel i-ésimo. Los super índices denotan: C - color, P - posición 3d y N - vector normal.
$f_g$ es una gaussiana y $f_w$ es una distribución watson.

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
[Link](DepthWMM.md)

## Distribución Gaussiana
Para un vector aleatorio $x=\left[ x_1, x_2, ...,x_d \right]$ de dimensión $d$. La gaussiana multivariada se define como:
$$f_g(x | u, \Sigma) = \dfrac{1}{(2\pi)^(d/2) det(\Sigma)^{1/2}} \exp\left( -\frac{1}{2} (x-u)^T \Sigma^{-1} (x-u ) \right)$$
Para escribir la gaussiana con la forma de la familia de exponenciales hay que definir:
* $t(x) = (x, -xx^t)$ como el estadístico suficiente.
* k(x) = =
* El parámetro de expetación $n=(\phi, \Phi) = (u, -(\Sigma + u u^T))$

## Divergencia de Bregman en la gaussiana

$$G_g(n) = -\frac{1}{2} \log (1 + \phi^T \Phi^{-1} \phi) - \frac{1}{2} log(\det(\Phi)) - \frac{d}{2} \log(2 \pi e ) $$

# Ejemplo de uso
```julia
using RGBDSegmentation
using FITSIO
using SegmentationBenchmark
using  PyPlot
using Plots
glvisualize()
using Images
using MAT
import RGBDSegmentation:JCSA
dataset = NYUDataset();
img     = get_image(dataset,419, compute_normals= true);
LAB     = rgb2lab(img.RGB);
config  = JCSA.Config(max_iterations = 500);
k       = 20;
labels  = clusterize(config,LAB, img.depth, img.normals, k)
imshow(labels)
```
