using Random
using LinearAlgebra
using MLDataPattern
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs, params
using Base.Iterators: repeated
using Printf
using DataFrames
using CSV
using ProgressMeter
using JLD
using Flux
using ScikitLearn

bithdv(N::Int=10000) = bitrand(N)

function bitadd(vectors::BitVector ...)
    v = reduce(.+, vectors)
    n = length(vectors) / 2
    x = [i > n ? 1 : i < n ? 0 : rand(0:1) for i in v]
    return convert(BitVector, x)
end

bitbind(vectors::BitVector ...) =  reduce(.⊻, vectors)

bitperm(vector::BitVector, k::Int=1) = circshift(vector, k)

hamming(x::BitVector, y::BitVector) = sum(x .!= y)/length(x)

rate::Float64 = ARGS[1]    # learning rate
batchsize::Int = ARGS[2]   # batch size
epochs::Int = ARGS[3]      # number of epochs
percent::Float64 = ARGS[4]

println("start netsurf")
netsurf = CSV.read("../data/netsurf.csv", DataFrame) # read dataset
select!(netsurf, Not(" cb513_mask")) # remove column
for i in ["input", " dssp3", " dssp8"] # remove white spaces in strings
    netsurf[!, i] = [join(map(x -> isspace(a[x]) ? "" : a[x], 1:length(a))) for a in netsurf[!, i]]
end
seq_list = netsurf.input
println("netsurf loaded")

println("load data")
x = vcat(
        JLD.load("data_netsurf/k1000.jld")["k25[ranges[i]:ranges[i + 1]]"], 
        [JLD.load(@sprintf("data_netsurf/k100%s.jld", i))["k25[ranges[i] + 1:ranges[i + 1]]"] for i in 1:52]...,
        JLD.load(@sprintf("data_netsurf/k100%s.jld", 53))["k25[ranges[i] + 1:length(k25)]"])
xl = length(x)

less_data = Random.sample(1:xl, round(percent*xl), replace = false)



