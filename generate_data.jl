# using Pkg
# Pkg.add("CSV")
# Pkg.add("Plots")
# Pkg.add("DataFrames")
# Pkg.add("StaticArrays")
# Pkg.add("DifferentialEquations")

# run GR backend as headless
ENV["GKSwstype"] = "100"

using CSV
using Plots
using DataFrames
using StaticArrays
using DifferentialEquations

# config
OUTPUT_PATH = "measurements.csv"

# use the GR backend for plots
gr()

# system of ODEs
function ode!(u, p, t)
    s = 0.5
    x, y = u
    a1, a2, b0, b1, b2 = p

    dx = (b0 * s) - (a1 * x) - (b1 * y)
    dy = (b2 * x) - (a2 * y)

    @SVector [dx, dy]
end

# true parameter values
params = (0.1, 0.2, 0.2, 0.6, 0.6)

# time span
tspan = (0.0, 50.0)

# input signals
signals = [0.5, 1.0, 2.0]

# initial conditions
u0 = @SVector [1.0, 0.0]

# set up problem
problem = ODEProblem(ode!, u0, tspan, params)

# obtain solution
sol = solve(problem, saveat=1.0)
sol_x = sol[1,:]
sol_y = sol[2,:]

# plot solution (& save plot)
plot(sol.t, sol_x, label="x", seriestype=:scatter)
plot!(sol.t, sol_y, label="y", seriestype=:scatter)
xlabel!("t")
ylabel!("activity")
savefig("sol.png")

# construct dataframe
df = DataFrame(time=sol.t, x=sol_x, y=sol_y)
println(df)

# save as csv
CSV.write(OUTPUT_PATH, df)
println("Saved to $OUTPUT_PATH")
