using Hadamard
using Random
using TimerOutputs
using LinearAlgebra
using Dates

"""
Compute the Subsampled Randomized Hadamard Transform of A with a sampling size of l.
Return B = Ω × A where A is m×n, Ω is l×m and B is lxn
Omega = sqrt(m/l) Dr R H Dl
* Dl and Dr are diagonal matrices of random signs
* H is the Hadamard matrix
* R is a sampling matrix, selecting l rows out of m

A should be stored in column major format.

In the context of a block HRHT: (Ω_1 ... Ω_p) (A_1 ... A_p)^t, then the
random number generator seeds should be passed to all hrht() calls.
param global_seed: same for all block
param block_seed: different for each block
"""
function hrht(A::Array{Float64,2}, l::Int; to::TimerOutput=TimerOutput(), global_seed::UInt=rand(UInt), block_seed::UInt=rand(UInt))::Array{Float64,2}
  rng_global = MersenneTwister(global_seed)
  rng_block = MersenneTwister(block_seed)
  #@info "$(now()) HRHT $(size(A)) into $l"; flush(stdout); flush(stderr)
  @assert l <= size(A)[1]
  @timeit to "init rand" begin
    # Rademacher vector
    Dr::Array{Int} = rand(rng_block, (-1,1), size(A)[1])
    Dl::Array{Int} = rand(rng_block, (-1,1), l)

    # Rescaling
    scale::Float64 = size(A)[1]/sqrt(l)
  end

  # X = Dr A
  @timeit to "Dr" lmul!(Diagonal(Dr), A)

  # X1 = H X
  @timeit to "H" fwht_natural!(A, 1)

  # 1-hashing
  @timeit to "1-hashing" begin 

    ## Associate each rows of A to an independent random number between 1 and l
    Pairs = [rand(rng_global,1:l) => c[:] for c in eachrow(A)]

    ## Filter each rows of A mapped associated to the same value, multiply them by a rademacher random variable
    ## And reduce by summing them. If there are no rows associated to a value, create a 0 row.
    ## Stack those rows and take the transpose (if use vcat, obtain a long vector which has to be reshaped)
    B::Array{Float64,2} = scale*hcat([mapreduce(p -> p.second*rand(rng_global,(-1,1)),+,filter(p -> p.first==i, Pairs),init=zeros(Int64,size(A)[2])) for i in 1:l]...)'

  end
  # Compute B = R Dl H Dr A
  @timeit to "Dl" lmul!(Diagonal(Dl), B)
  
  #@info "$(now()) HRHT done"; flush(stdout); flush(stderr)
  return B
end
