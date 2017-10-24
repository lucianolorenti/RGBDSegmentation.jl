using Plots
function plot_normals(img)

colors=vec([RGB((img.RGB[v[1],v[2],:]/255)...) for v in CartesianRange((size(img.RGB,1),size(img.RGB,2))) ])
z=vec(img.depth[:,:,3])
y=vec(img.depth[:,:,2])
x=vec(img.depth[:,:,1])
line_points = zeros(size(img.RGB,1)*size(img.RGB,2)*3, 3)
j = 1
r = 1
for v in CartesianRange((size(img.RGB,1),size(img.RGB,2)))
    normal = img.normals[v[1],v[2],1:3]
    pos    =[x[j]; y[j];z[j]]
    line_points[r,:] = pos - 0.05*normal
    r=r+1
    line_points[r,:] = pos + 0.05*normal
    r=r+1
    line_points[r,:] = [NaN;NaN;NaN]
    r=r+1
    j=j+1
end
glvisualize()
plt = Plots.plot(line_points[:,1],line_points[:,2], line_points[:,3], linewidth=Float32(0.5))
plt = Plots.scatter!(plt,x,y,z,color=colors, markersize=0.9, markerstrokewidth = 0)
end
