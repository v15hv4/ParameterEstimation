using CSV, Catalyst, Plots, DataFrames, DifferentialEquations, Random, Distributions

# config
MEASUREMENTS_PATH = "measurements.csv"

# reaction network
rn = @reaction_network begin
    α_1, X --> ∅
    α_2, Y --> ∅
    β_0 * S, ∅ --> X
    -β_1 * Y, ∅ --> X
    β_2 * X, ∅ --> Y
end α_1 α_2 β_0 β_1 β_2

# timespan of experiment
tspan = (0.0, 50.0)
teval = (tspan[2] - tspan[1]) / 1000

# nominal parameters
p = (
    :α_1 => 0.1,
    :α_2 => 0.2,
    :β_0 => 0.2,
    :β_1 => 0.6,
    :β_2 => 0.6,
)

# initial conditions
u0 = [
    :X => 1.0,
    :Y => 0.0,
    :S => 0.5,
]

# define ODE problem
problem = ODEProblem(rn, u0, tspan, p)

# solve the ODE
sol = solve(problem, Tsit5(), saveat=teval)
sol_X = sol[1, :]
sol_Y = sol[2, :]

# add noise to observations
add_noise = rand(Normal(0.0, 0.05), size(sol_X))
mul_noise = rand(LogNormal(0.0, 0.05), size(sol_X))
output = (sol_X .* mul_noise) .+ add_noise

# plot solution (& save plot)
plot(sol.t, output, color="black", alpha=0.3, label="output", seriestype=:scatter)
plot!(sol.t, sol_X, color="blue", label="X(t)")
plot!(sol.t, sol_Y, color="red", label="Y(t)")
xlabel!("t")
ylabel!("activity")
savefig("sol.png")

# construct dataframe
df = DataFrame(time=sol.t, X=sol_X, Y=sol_Y, output=output)
println(df)

# save as csv
CSV.write(MEASUREMENTS_PATH, df)
println("Saved to $MEASUREMENTS_PATH")
