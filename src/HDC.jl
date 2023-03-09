using Random
using LinearAlgebra
"""
Construct a binary vector. By default 10000 elements long.
"""
bithdv(N::Int=10000) = bitrand(N)
"""
Bundles binary hyperdimensional vectors based on the element-wise majority rule.
"""
function bitadd(vectors::BitVector ...)
    v = reduce(.+, vectors)
    n = length(vectors) / 2
    x = [i > n ? 1 : i < n ? 0 : rand(0:1) for i in v]
    return convert(BitVector, x)
end
"""
Binds binary hyperdimensional vectors based on an element-wise XOR gate.
"""
bitbind(vectors::BitVector ...) =  reduce(.⊻, vectors)
"""
Permutes a binary hyperdimensional vector by an adjustable circular shift.
"""
bitperm(vector::BitVector, k::Int=1) = circshift(vector, k)
"""
Calculates the Hamming distance between two binary vectors.
"""
hamming(x::BitVector, y::BitVector) = sum(x .!= y)/length(x)
"""
Construct a bipolar vector. By default 10000 elements long.
"""
hdv(N::Int=10000) = rand!(zeros(N), (-1,1))
"""
Bundles bipolar hyperdimensional vectors.
"""
add(vectors...) = reduce(.+, vectors) .|> sign
"""
Binds binpolar hyperdimensional vectors.
"""
bind(vectors...) = reduce(.*, vectors)
"""
Permutes a bipolar hyperdimensional vector by an adjustable circular shift.
"""
perm(vector::Vector, k::Int=1) = circshift(vector, (0, k))
"""
Calculates the cosine similarity between two bipolar vectors.
"""
cosine(x::Vector, y::Vector) = dot(x, y) / (norm(x) * norm(y))
"""
Creates a list of bipolar hyperdimensional vectors representing an interval of numbers by constructing a random HDV representing the lower bound of the interval and replacing a fraction of the vector with random bits.
"""
function range_hdvs(steps)
    k = length(steps) - 1
    V = [hdv() for i in 1:k+1]
    for i in 2:k+1
        for j in 1:10000
            V[i][j] = rand((-1, 1))
        end
    end
    return V
end
"""
Simple 2-layer convolved embedding in hyperdimensional space
"""
function convolved_embedding(sequence, tokens, k=3)
    # layer 1
    kmer_hdvs = []
    for i in 1:length(sequence)-k+1
        kmer = sequence[i:i+k-1]
        aa_hdvs = [circshift(tokens[aa], k-l) for (l, aa) in enumerate(kmer)]
        push!(kmer_hdvs, bitbind(hcat(aa_hdvs)...))
    end
    
    # layer 2
    conv_kmer_hdvs = []
    for i in 1:length(kmer_hdvs)-k+1
        convolved_kmers = kmer_hdvs[i:i+k-1]
        conv_hdvs = [circshift(convolved_kmers[l], k-l) for (l, km) in enumerate(convolved_kmers)]
        push!(conv_kmer_hdvs, bitbind(hcat(conv_hdvs)...))
    end
    
    return bitadd(hcat(conv_kmer_hdvs)...)
end