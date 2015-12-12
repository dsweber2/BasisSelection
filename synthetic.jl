include("ksvd.jl");
k=20
n = 5
m = 50
oldDict = randn(n,k)
for i=1:k
  oldDict[:,i] = oldDict[:,i]/norm(oldDict[:,i],2)
end
data= zeros(n,m)
for i=1:m
  data[:,i] = 30*oldDict[:,rand(1:k,3)]*rand(3,1)
end
maxIter = 10
peps = .5
pMaxIter = 11; pursuitAlgorithm=1
include("ksvd.jl"); (minn,dict,x)= ksvd(data, k, maxIter, peps, pMaxIter, pursuitAlgorithm)
x.value
sparsity(x.value)
include("ksvd.jl");
tes = [1 2 1e-6 1e-5; 2 1e-2 4 1e-7]
sparsity(tes)
vecnorm(data)/n/m
norm(dict-oldDict)/norm(oldDict)

data[:,1]-dict*x.value[:,1]
