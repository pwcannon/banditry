using Distributions
using Plots

mutable struct Player 
    actionValues::Vector{Float64}
    totalReward::Float64
    lastReward::Float64
    roundsPlayed::Vector{Int}
end

struct Bandit
    arms::Vector{Distribution}
end

function Player(b::Bandit)::Player
    Player([abs(rand(Normal(0.0, 0.001))) for i=1:length(b.arms)], 0.0, 0.0, [0 for i=1:length(b.arms)])
end

function pull(b::Bandit, arm::Int)::Float64
    rand(b.arms[arm])
end

function choosearm(p::Player, ϵ::Float64)::Int
    # ϵ-greedy strategy
    if rand() < ϵ
        arm = rand(1:length(p.actionValues))
    else
        arm = findmax(p.actionValues)[2]
    end
    arm
end

function play!(p::Player, b::Bandit, ϵ::Float64)
    arm = choosearm(p, ϵ)
    reward = pull(b, arm)
    p.totalReward += reward
    p.lastReward = reward
    p.roundsPlayed[arm] += 1
    p.actionValues[arm] += (reward - p.actionValues[arm])/p.roundsPlayed[arm] 
    p
end

function Base.rand(b::Bandit, nsteps::Int, ϵ)
    total = zeros(nsteps)
    army = zeros(nsteps)
    p = Player(b)
    for i = 1:nsteps
        arm = play!(p, b, ϵ)
        total[i] = p.totalReward
        army[i] = arm
    end
    return(total, army)
end

function Base.rand(b::Bandit, nsteps::Int, nplayers::Int, ϵ::Float64)
    total = zeros(nsteps, nplayers)
    for i = 1:nplayers
        p = Player(b)
        for j = 1:nsteps
            play!(p, b, ϵ)
            total[j, i] = p.lastReward
        end
    end
    total
end

function mysamp(nBandits::Int, sizeproblemset::Int, nsteps::Int, epsVec::Vector{Float64})
    result = zeros(nsteps, length(epsVec))
    for i = 1:sizeproblemset
        banditMeans = rand(Normal(0,1), nBandits)
        b = Bandit([Normal(banditMeans[i], 1.0) for i=1:nBandits])
        for j = 1:length(epsVec)
            result[:,j] += rand(b, nsteps, 1, epsVec[j])
        end
    end
    result /= sizeproblemset
end

theme(:wong2)
meanruns = mysamp(10, Int(1e4), 1000, [0.0,0.01,0.1]);
plot(1:1000, meanruns, label = ["eps = 0", "eps = 0.01", "eps = 0.1"], 
        lw=2, ylims=(0,1.5), xlims=(0,1000), xlabel = "Steps", ylabel = "Average reward", legend=:bottomright)
savefig("/Users/patrickcannon/src/github.com/pwcannon/bandit-experiments/10-armBandit2.pdf")