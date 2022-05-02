using CSV, Catalyst, Plots, DataFrames, DifferentialEquations

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
sol = solve(problem, Tsit5(), saveat=1.0)
sol_X = sol[1,:]
sol_Y = sol[2,:]

# plot solution (& save plot)
plot(sol.t, sol_X, label="X(t)", seriestype=:scatter)
plot!(sol.t, sol_Y, label="Y(t)", seriestype=:scatter)
xlabel!("t")
ylabel!("activity")
savefig("sol.png")

# construct dataframe
df = DataFrame(time=sol.t, X=sol_X, Y=sol_Y)
println(df)

# save as csv
CSV.write(MEASUREMENTS_PATH, df)
println("Saved to $MEASUREMENTS_PATH")
