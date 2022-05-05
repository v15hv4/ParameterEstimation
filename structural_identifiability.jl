using StructuralIdentifiability

# Given x1(t), infer all parameters
ode = @ODEmodel(
    x1'(t) = (b0 * s(t)) - (a1 * x1(t)) - (b1 * x2(t)),
    x2'(t) = (b2 * x1(t)) - (a2 * x2(t)),
    y(t) = x1(t)
)

# Given x1(t) and b1, infer all other parameters
ode_b1 = @ODEmodel(
    x1'(t) = (b0 * s(t)) - (a1 * x1(t)) - ((6 / 10) * x2(t)),
    x2'(t) = (b2 * x1(t)) - (a2 * x2(t)),
    y(t) = x1(t)
)

# Given x1(t) and a1, infer all other parameters
ode_a1 = @ODEmodel(
    x1'(t) = (b0 * s(t)) - ((1 / 10) * x1(t)) - (b1 * x2(t)),
    x2'(t) = (b2 * x1(t)) - (a2 * x2(t)),
    y(t) = x1(t)
)

id_res = assess_identifiability(ode)
id_res_b1 = assess_identifiability(ode_b1)
id_res_a1 = assess_identifiability(ode_a1)