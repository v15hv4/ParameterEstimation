include("./NFBSystem.jl")
import .NFBSystem as System

using DifferentialEquations, DiffEqSensitivity, LinearAlgebra
using Plots, Statistics

function rn(du, u, p, t)
    X, Y, S = u
    α_1, α_2, β_0, β_1, β_2 = p
    du[1] = (β_0 * S) - (α_1 * X) - (0.6 * Y)
    du[2] = (β_2 * X) - (α_2 * Y)
    du[3] = 0
end

params = [0.082, 0.221, 0.188, 0.6, 0.613]
u0 = [1.0; 0.0; 0.5]
problem = ODEForwardSensitivityProblem(rn, u0, System.tspan, params)
sol = solve(problem, Tsit5(), saveat=System.teval)

x, dp = extract_local_sensitivities(sol)

sigma = 0.05 * std(x, dims=2)
cov_error = sigma[1]

cols = 1:1
Nt = length(dp[1][1, :])
Nstate = length(dp[1][:, 1])
Nparam = length(dp[:, 1])
FIM = zeros(Float64, Nparam, Nparam)

perm = vcat(1, sort(rand(2:Nt-1, Nt ÷ 5)), Nt)

for i in perm
    S = reshape(dp[1][:, i], (Nstate, 1))
    for j = 2:Nparam
        S = hcat(S, reshape(dp[j][:, i], (Nstate, 1)))
    end
    global FIM += S[cols, :]' * inv(cov_error) * S[cols, :]
end

FIM = FIM[1:end.!=4, 1:end.!=4]

C = inv(FIM)
R = ones(size(C))

R = [C[i, j] / sqrt(C[i, i] * C[j, j]) for i = 1:size(C)[1], j = 1:size(C)[1]]

lab = ["α_1", "α_2", "β_0", "β_2"]

heatmap(R, aspect_ratio=1, color=:acton,
    clims=(-1, 1),
    xlims=(0.5, size(R)[1] + 0.5),
    xticks=(1:1:size(C)[1], lab),
    yticks=(1:1:size(C)[1], lab),
)
savefig("correlation_matrix.png")

abs.(R) .> 0.99

tscale = 1.0
cscale = 1.0
lowerbound = sqrt.(diag(inv(FIM))) / tscale
for i = 1:length(lab)
    println(lab[i], '\t', lowerbound[i])
end

for i = 1:(Nparam-1)
    println(eigvals(FIM)[i])
    println(eigvecs(FIM)[:, i])
    println('\n')
end

plot(eigvals(FIM), seriestype=:scatter, yaxis=:log, xticks=(1:1:size(lab)[1], lab))
savefig("eigenvalues.png")

bar(eigvecs(FIM)[:, 1], ylabel="FIM null eigenvector coefficients", ytickfont=font(12, "Times"),
    xticks=(1:1:size(C)[1], lab), xtickfont=font(12, "Courier"),
    legendfontsize=10,
    legend=:topright)
savefig("null_eigenvector.png")