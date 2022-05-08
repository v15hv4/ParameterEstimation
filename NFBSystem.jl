module NFBSystem

using Catalyst, OrderedCollections

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
nominal_parameters = OrderedDict(
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

end