include("./NFBSystem.jl")
import .NFBSystem as System

using CSV, Plots, DataFrames
using DifferentialEquations, Random, Distributions

# config
MEASUREMENTS_PATH = "measurements.csv"

# extract and arrange nominal parameters
p = collect(values(System.nominal_parameters))

# define ODE problem
problem = ODEProblem(System.rn, System.u0, System.tspan, p)

# solve the ODE
sol = solve(problem, Tsit5(), saveat=System.teval)
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