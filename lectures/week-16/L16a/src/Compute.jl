# PRIVATE METHODS BELOW HERE ================================================================================= #
function _world(model::MyRectangularGridWorldModel, s::Int, a::Int)::Tuple{Int,Float64}
    model, a # dummy default; replace with a problem-specific worldmodel kwarg in solve(...)
    return (s, 0.0);
end
# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #
"""
    solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel;
        maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world) -> MyQLearningAgentModel

Simulate the Q-Learning agent in the given environment for a maximum number of steps per starting state.

### Arguments
- `agent::MyQLearningAgentModel`: the Q-Learning agent model.
- `environment::MyRectangularGridWorldModel`: the environment model.
- `maxsteps::Int = 100`: the maximum number of steps to simulate from each starting state.
- `δ::Float64 = 0.02`: the convergence threshold on the change in `Q` between iterations.
- `worldmodel::Function = _world`: function `(environment, s, a) -> (s′, r)` that returns the next state and reward.

### Returns
- `MyQLearningAgentModel`: the updated Q-Learning agent model after simulation.
"""
function solve(agent::MyQLearningAgentModel, environment::MyRectangularGridWorldModel;
    maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world)::MyQLearningAgentModel

    # initialize -
    actions = agent.actions;
    K = length(actions); # number of actions
    states = agent.states;
    Q₁ = agent.Q;
    γ = agent.γ;

    # simulation loop -
    for s ∈ states

        # initialize t -
        t = 1;
        has_converged = false;
        αₜ = copy(agent.α);
        while (has_converged == false)

            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3);
            p = rand();

            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K); # generate a random action
            elseif p > ϵₜ
                aₜ = argmax(Q₁[s,:]); # select the greedy action, given state s
            end

            # compute new state and reward -
            s′, r = worldmodel(environment, s, aₜ);

            # use the update rule to update Q -
            Q₂ = copy(Q₁);
            Q₁[s,aₜ] += αₜ*(r + γ*maximum(Q₁[s′,:]) - Q₁[s,aₜ])

            # update stuff -
            s = s′; # state update
            t += 1; # time update
            αₜ = 0.99*αₜ; # update the learning rate

            # check if we have converged -
            if ((t > maxsteps) || norm(Q₂ - Q₁) < δ)
                has_converged = true;
            end
        end
    end

    agent.Q = Q₁; # update the model

    # return -
    return agent
end

"""
    policy(Q_array::Array{Float64,2}) -> Array{Int,1}

Compute the greedy policy from a Q-value table.

### Arguments
- `Q_array::Array{Float64,2}`: the Q-value table with rows indexed by state and columns by action.

### Returns
- `Array{Int,1}`: the policy mapping each state to the action with the largest Q-value.
"""
function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    (NR, _) = size(Q_array);

    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    return π_array;
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #
