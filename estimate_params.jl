using CSV
using Plots
using Statistics
using DataFrames
using DiffEqFlux
using DifferentialEquations

# run GR backend as headless
ENV["GKSwstype"] = "100"

# config
DATASET_PATH = "measurements.csv"

# use the GR backend for plots
gr()

# ode system
function ode!(du, u, p, t)
    s = 0.5
    x, y = u

    a2, b0, b2 = p
    a1 = 0.1
    b1 = 0.6

    du[1] = (b0 * s) - (a1 * x) - (b1 * y)
    du[2] = (b2 * x) - (a2 * y)
end

# initial conditions
u0 = [1.0, 0.0]

# simulation interval & evaluation points
tspan = (0.0, 50.0)

# initial parameters (b1 is fixed)
p = [0.0, 0.0, 0.0]

# set up problem
problem = ODEProblem(ode!, u0, tspan, p)

# # obtain solution
# sol = solve(problem, Tsit5())
# 
# # plot solution
# plot(sol)
# savefig("estim_sol.png")

# load dataset
df = DataFrame(CSV.File(DATASET_PATH))

# loss function
function loss(p)
    sol = solve(problem, Tsit5(), p=p, saveat=1.0)
    loss = mean((sol[1,:] - df.x) .^ 2)
    return loss, sol
end

# nn callback
callback = function (p, l, pred)
    println("loss: $l - parameters: $p")
    plt = plot(pred, linewidth=2)
    savefig("prediction.png")
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

# optimize
result_ode = DiffEqFlux.sciml_train(loss, p, cb=callback, maxiters=100)
