# using Pkg
# Pkg.add("CSV")
# Pkg.add("Plots")
# Pkg.add("Statistics")
# Pkg.add("DataFrames")
# Pkg.add("DiffEqFlux")
# Pkg.add("DifferentialEquations")

using CSV, Plots, Statistics, DataFrames, DiffEqFlux, DifferentialEquations

# config
DATASET_PATH = "measurements.csv"

# ode system
function ode!(du, u, p, t)
    s = 0.5
    x, y = u

    a1 = 0.1
    b1 = 0.6
    a2, b0, b2 = p

    du[1] = (b0 * s) - (a1 * x) - (b1 * y)
    du[2] = (b2 * x) - (a2 * y)
end

# initial conditions
u0 = [1.0, 0.0]

# simulation interval and evaluation points
tspan = (0.0, 50.0)
tint = 1.0

# initial parameter values
p = [0.0, 0.0, 0.0]

# set up ode problem
problem = ODEProblem(ode!, u0, tspan, p)

# load dataset
df = DataFrame(CSV.File(DATASET_PATH))
actual_x = Array(df.x)

# loss function
function loss(p)
    sol = solve(problem, Tsit5(), p=p, saveat=tint)
    pred_x = sol[1, :]
    loss = mean((pred_x - actual_x) .^ 2)
    return loss, sol
end

# nn callback
callback = function (p, l, pred)
    display("loss: $l -- params: $p")
    plt = plot(pred)
    title!("a2: $(round.(p[1]; digits=3)), b0: $(round.(p[2]; digits=3)), b2: $(round.(p[3]; digits=3))")
    display(plt)
    return false
end

result_ode = DiffEqFlux.sciml_train(loss, p, cb=callback)
savefig("estimation_results.png")