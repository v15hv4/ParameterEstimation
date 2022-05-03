using CSV, Catalyst, Plots, DataFrames, DifferentialEquations, Evolutionary, Statistics

# config
MEASUREMENTS_PATH = "measurements.csv"

# reaction network
rn = @reaction_network begin
    α_1, X --> ∅
    α_2, Y --> ∅
    β_0 * S, ∅ --> X
    -0.6 * Y, ∅ --> X
    β_2 * X, ∅ --> Y
end α_1 α_2 β_0 β_2

# timespan of experiment
tspan = (0.0, 50.0)

# initial conditions
u0 = [
    :X => 1.0,
    :Y => 0.0,
    :S => 0.5,
]

# load measurements
df = DataFrame(CSV.File(MEASUREMENTS_PATH))
actual = Array(df.X)

# objective function to optimize
function objective(p)
    problem = ODEProblem(rn, u0, tspan, p)
    sol = solve(problem, Tsit5(), saveat=1.0)
    preds = sol[1, :]
    mse = mean((preds - actual) .^ 2)

    plt = plot(sol, linewidth=2)
    title!("mse: $mse")
    display(plt)

    return mse
end

# initial parameters
p0 = zeros(4)

# parameter bounds
lb = zeros(4)
ub = ones(4)
bounds = BoxConstraints(lb, ub)

# algorithm
alg = CMAES()

# optimizer options
opts = Evolutionary.Options(
    parallelization=:thread,
)

# run evolutionary optimization using the selected algorithm
res = Evolutionary.optimize(objective, bounds, p0, alg, opts)

# extract estimated parameters
pe = round.(Evolutionary.minimizer(res); digits=3)
println("Estimated parameters: $pe")