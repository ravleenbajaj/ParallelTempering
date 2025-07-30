begin
	using CSV, DataFrames, Turing, Distributions, FillArrays, LinearAlgebra
	using Pigeons, MCMCChains, StatsPlots
	
	# Load a smaller spin configuration matrix
	X_df = CSV.read("/Users/ravleenbajaj/X.csv", DataFrame)
	X_matrix = Int.(Matrix(X_df[1:20, 1:10]))  # ↓ Debug with fewer samples & spins

	# Define the Ising model
	@model function IsingTuring(X; σ_prior=1.0)
	    n_samples, n_spins = size(X)
	    n_params = n_spins*(n_spins-1) ÷ 2
	    J_upper ~ filldist(Normal(0, σ_prior), n_params)

	    J = zeros(eltype(J_upper), n_spins, n_spins)
	    idx = 1
	    for i in 1:n_spins
	        for j in i+1:n_spins
	            J[i,j] = J_upper[idx]
	            J[j,i] = J_upper[idx]
	            idx += 1
	        end
	    end

	    for n in 1:n_samples
	        x = convert.(eltype(J), X[n, :])
	        energy = dot(x, J * x)  # ✅ Vectorized energy
	        Turing.@addlogprob! energy
	    end
	end

	# Wrap the model
	tp = TuringLogPotential(IsingTuring(X_matrix, σ_prior=1.0))

	# Run Pigeons with RandomWalk
	pt = pigeons(
	    target = tp,
	    n_chains = 5,
	    n_rounds = 6,
	    record = [traces, round_trip],
	    explorer = SliceSampler()  # ✅ Much faster
	)

	# Convert and plot
	samples = Chains(pt)
	plot(samples)
	savefig("ising_pt_traceplots.html")
end
