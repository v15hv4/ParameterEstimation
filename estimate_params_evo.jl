using Catalyst, DifferentialEquations, Evolutionary
using CSV, DataFrames, Plots, Printf, Statistics
using Wandb, Logging

# config
MEASUREMENTS_PATH = "measurements.csv"
WANDB_PROJECT = "Evolutionary_ParameterEstimation"

# wandb
lg = WandbLogger(project=WANDB_PROJECT, name=nothing)
global_logger(lg)

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
teval = (tspan[2] - tspan[1]) / 1000

# nominal parameters
nominal_parameters = [
    0.1, # α_1
    0.2, # α_2
    0.2, # β_0
    # 0.6, # β_1
    0.6, # β_2
]

# initial conditions
u0 = [
    :X => 1.0,
    :Y => 0.0,
    :S => 0.5,
]

# load measurements
println("Loading dataset...")
df = DataFrame(CSV.File(MEASUREMENTS_PATH))
actual = Array(df.output)

# objective function to optimize
function objective(p)
    problem = ODEProblem(rn, u0, tspan, p)
    sol = solve(problem, Tsit5(), saveat=teval)
    preds = sol[1, :]
    mse = mean((preds - actual) .^ 2)

    # show current fit
    plt = plot(sol, linewidth=2)
    mse_str = @sprintf("%.4E", mse)
    title!("mse: $mse_str")
    display(plt)

    return mse
end

# callback to log loss on wandb at every iteration
function wandb_loss_cb(trace)
    # log mse
    @info "metrics" loss = trace.value

    return false
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
    callback=wandb_loss_cb,
)

# track config
config = Dict(
    "algorithm" => String(Symbol(alg)),
    "bounds" => (lb, ub),
    "init_params" => p0,
)
update_config!(lg, config)

# run evolutionary optimization using the selected algorithm
println("Running evolutionary optimization...")
res = Evolutionary.optimize(objective, bounds, p0, alg, opts)

# extract estimated parameters
estimated_parameters = round.(Evolutionary.minimizer(res); digits=3)
println("Estimated parameters: $estimated_parameters")
println("Nominal parameters: $nominal_parameters")

# plot fit for logging
problem = ODEProblem(rn, u0, tspan, estimated_parameters)
sol = solve(problem, Tsit5(), saveat=teval)
sol_X = sol[1, :]
sol_Y = sol[2, :]
actual_X = Array(df.X)
actual_Y = Array(df.Y)
plt = plot(actual, color="black", alpha=0.2, label="X (data)", seriestype=:scatter)
plot!(actual_X, color="blue", linewidth=2, label="X (actual)", linestyle=:dash)
plot!(actual_Y, color="red", linewidth=2, label="Y (actual)", linestyle=:dash)
plot!(sol_X, color="blue", linewidth=2, label="X (pred)")
plot!(sol_Y, color="red", linewidth=2, label="Y (pred)")
xlabel!("t")
ylabel!("activity")

# log run results
println("Logging results on WandB...")
parameter_table = Wandb.wandb.Table(
    data=[estimated_parameters - nominal_parameters],
    columns=["α_1", "α_2", "β_0", "β_2"]
)
Wandb.log(
    lg,
    Dict(
        "results/iters" => Evolutionary.iterations(res),
        "results/parameter_error" => parameter_table,
        "results/fit" => Wandb.Image(plt),
        "results/parameter_mse" => mean((nominal_parameters - estimated_parameters) .^ 2),
    )
)

# finish wandb run
close(lg)