include("./NFBSystem.jl")
import .NFBSystem as System

using DifferentialEquations, Evolutionary
using CSV, DataFrames, Plots, Printf, Statistics
using Wandb, Logging

# config
MEASUREMENTS_PATH = "measurements.csv"
WANDB_PROJECT = "EvolutionaryParameterEstimation"

# wandb
println("Initializing WandB...")
lg = WandbLogger(project=WANDB_PROJECT, name=nothing)
global_logger(lg)

# load measurements
println("Loading dataset...")
df = DataFrame(CSV.File(MEASUREMENTS_PATH))
actual = Array(df.output)

# subset of parameters to estimate
target_params = [:α_1, :α_2, :β_0, :β_2]
println("Target parameters: $target_params")

# function to extract and order parameters
function make_params(param_values)
    param_dict = Dict(zip(target_params, param_values))
    ordered_params = [get(param_dict, i, System.nominal_parameters[i]) for i in keys(System.nominal_parameters)]
    return ordered_params
end

# objective function to optimize
function objective(p)
    # define problem and solve
    problem = ODEProblem(System.rn, System.u0, System.tspan, make_params(p))
    sol = solve(problem, Tsit5(), saveat=System.teval)
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
p0 = zeros(length(target_params))

# parameter bounds
lb = zeros(length(target_params))
ub = ones(length(target_params))
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
nominal_parameters = [System.nominal_parameters[i] for i in target_params]
println("Estimated parameters: $estimated_parameters")
println("Nominal parameters: $nominal_parameters")

# plot fit for logging
problem = ODEProblem(System.rn, System.u0, System.tspan, make_params(estimated_parameters))
sol = solve(problem, Tsit5(), saveat=System.teval)
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
    data=[estimated_parameters],
    columns=map(String, target_params)
)
parameter_error_table = Wandb.wandb.Table(
    data=[estimated_parameters - nominal_parameters],
    columns=map(String, target_params)
)
Wandb.log(
    lg,
    Dict(
        "results/iters" => Evolutionary.iterations(res),
        "results/fit" => Wandb.Image(plt),
        "results/parameters" => parameter_table,
        "results/parameter_error" => parameter_error_table,
        "results/parameter_mse" => mean((nominal_parameters - estimated_parameters) .^ 2),
        "results/parameter_error_percentage" => mean((nominal_parameters - estimated_parameters) ./ nominal_parameters) * 100,
    )
)

# finish wandb run
close(lg)