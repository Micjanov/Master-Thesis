using Random
using LinearAlgebra
"""
Instead of maximalizing the similarity of the addition between HDVs, can we do a partial addition for binary vectors?
coef must be smaller than 1. Only 2 vectors allowed and should be equal in length. Outputs the 1st vector that has been modified to make it more similar to the 2nd vector, extend of similarity maximalizing is adjustable via coef.
"""
function part_bitadd(vector1, vector2, coef=1)
    N = length(vector1)
    N2 = length(vector2)

    if N != N2
        throw(error("Vectors must be equal in size"))
    end

    n_indices = N* coef |> round
    v = copy(vector1)
    indices = [rand(1:N) for i in 1:n_indices]
    for i in indices
        v[i] = vector2[i] == 1 ? 1 : vector2[i] == 0 : 0
    end
    return v
end