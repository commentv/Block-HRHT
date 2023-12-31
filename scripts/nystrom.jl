using Dates

using LinearAlgebra
using TimerOutputs

LinearAlgebra.BLAS.set_num_threads(1)

include("../src/init_cluster.jl")

include("../src/gaussian_sampling.jl")
include("../src/rbs_kernel.jl")
include("../src/block_hrht.jl")
include("../src/io.jl")
include("../src/matrix_operations.jl")

println("include finished")
flush(stdout)

function main()

  @everywhere to = TimerOutput()

  filename::String = ENV["BHRHT_MATRIX_FILE"]
  σ = parse(Float64, ENV["BHRHT_SIGMA"])
  nb_exp = parse(Int, ENV["BHRHT_NBEXP"])
  ls::Vector{Int} = eval(Meta.parse(ENV["BHRHT_LS"]))
  ks::Vector{Int} = eval(Meta.parse(ENV["BHRHT_KS"]))
  preload::Bool = eval(Meta.parse(ENV["BHRHT_PRELOAD"]))
  sep::Char = eval(Meta.parse(ENV["BHRHT_SEP"]))
  println("$(now()) replacement = $(ENV["BHRHT_RPLC"])")

  A = generate_darray(filename, σ, sep=sep)

  trace_A = nuclear_norm(A)

  println("$(now()) input matrix size $(size(A))")
  println("$(now()) Sigma $σ")
  println("$(now()) l = $ls")
  println("$(now()) grid = $(size(procs(A)))")
  println("$(now()) preload = $preload")
  println("$(now()) nb_exp = $nb_exp")
  flush(stdout)

  if (preload)
    k = ks[1]
    l = ls[1]

    S_s,CDp_s, V_s, k_tol = sketch_nystrom(A, l, to)
    println("k_tol : $k_tol")
    S_g, CDp_g, V_g = sketch_nystrom_gaussian(A, l, to)

    res = [@spawnat p (CDp_s[:l] * V_s[:,1:k_tol] * Diagonal(1 ./ S_s[1:k_tol]))' for p in procs(CDp_s)]
    [wait(r) for r in res]
    QUk_s = DArray(reshape(res, (1,length(procs(CDp_s)))))
    B = multiply(diagm(S_s[1:k]), QUk_s)

    res = [@spawnat p (CDp_g[:l] * V_g[:,1:k] * Diagonal(1 ./ S_g[1:k]))' for p in procs(CDp_g)]
    [wait(r) for r in res]
    QUk_g = DArray(reshape(res, (1,length(procs(CDp_g)))))
    B_gaussian = multiply(diagm(S_g[1:k]), QUk_g)
  end

  matrixname = basename(filename)
  timings_csv = open("timings_$matrixname$(nworkers())_kmin$(ks[1])_height$(size(A)[1]).csv", "w")

  for l in ls
    error_avg = 0.0
    error_min = Inf
    error_max = 0.0
    for i in 1:nb_exp
      local UtAU, UtA, B
      @timeit to "hrht l=$l" S_s,CDp_s, V_s, k_tol = sketch_nystrom(A, l, to)
      println("k_tol : $k_tol")
      @timeit to "gaussian l=$l" S_g, CDp_g, V_g = sketch_nystrom_gaussian(A, l, to)
      for k in ks
        if k <= l
          file_gaussian = open("nuclear_error_gaussian_$(matrixname)" *
                               "_p$(nworkers())_l$(l)_k$(k)_sig$(σ)" *
                               "_height$(size(A)[1]).csv", "w")
          file_hrht = open("nuclear_error_hrht_$(matrixname)" * 
                           "_p$(nworkers())_l$(l)_k$(k)_sig$(σ)" *
                           "_height$(size(A)[1]).csv", "w")

          res = [@spawnat p (CDp_s[:l]
                             * V_s[:,1:k_tol]
                             * Diagonal(1 ./ S_s[1:k_tol]))'
                 for p in procs(CDp_s)]
          [wait(r) for r in res]
          QUk_s = DArray(reshape(res, (1,length(procs(CDp_s)))))
          B = multiply(diagm(S_s[1:k_tol]), QUk_s)

          res = [@spawnat p (CDp_g[:l]
                             * V_g[:,1:k]
                             * Diagonal(1 ./ S_g[1:k]))'
                 for p in procs(CDp_g)]
          [wait(r) for r in res]
          QUk_g = DArray(reshape(res, (1,length(procs(CDp_g)))))
          B_gaussian = multiply(diagm(S_g[1:k]), QUk_g)

          error_hrht = nuclear_norm(A - B) / trace_A
          error_gaussian = nuclear_norm(A - B_gaussian) / trace_A
          write(file_hrht, string(error_hrht) * "\n")
          write(file_gaussian, string(error_gaussian) * "\n")

          error_min = min(error_min, error_hrht / error_gaussian)
          error_max = max(error_max, error_hrht / error_gaussian)
          error_avg += error_hrht / error_gaussian
          @info "$(now()) l = $l in $ls (k_tol $(size(V_s[1]))) " *
                "and k = $k - Exp $i / $nb_exp"; flush(stdout)
          close(file_hrht)
          close(file_gaussian)
        end
      end
      close(CDp_s)
      close(CDp_g)
      GC.gc()
      [remotecall_wait(GC.gc, p) for p in workers()]
      println("Error l = $l min=", error_min, " avg=", error_avg / nb_exp, " max=", error_max)
      write(timings_csv, string(l) * " "
            * string((TimerOutputs.time(to["hrht l=$l"]["reduce UtA"]) +
                   TimerOutputs.time(to["hrht l=$l"]["reduce UtAU"])) / (nb_exp * 1e6)) * " "
            * string(TimerOutputs.time(to["hrht l=$l"]["tsqr"]) / (nb_exp * 1e6)) * " "
            * string(TimerOutputs.time(to["hrht l=$l"]["last step"]) / (nb_exp * 1e6)) * " "
            * string((TimerOutputs.time(to["gaussian l=$l"]["reduce UtA"]) +
                   TimerOutputs.time(to["gaussian l=$l"]["reduce UtAU"])) / (nb_exp * 1e6)) * "\n"
           )
    end
  end
  close(A)
  close(timings_csv)

  show(to)
end

main()
