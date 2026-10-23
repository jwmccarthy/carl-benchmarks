#include <RLGymPPO_CPP/Learner.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/DefaultOBS.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/StateSetters/KickoffState.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace RLGPC;
using namespace RLGSC;

namespace {
constexpr int kFrameskip         = 8;
constexpr int kCollection        = 100000;
constexpr int kMinibatch         = 25000;
int           threads            = 16;
int           gamesPerThread     = 24;
int           warmupIterations   = 1;
int           epochs             = 1;
std::ofstream logFile;
int           reportedIterations = 0;

EnvCreateResult CreateEnv() {
    auto rewards = new CombinedReward(
        {
            {new EventReward({.teamGoal = 10.f, .concede = -10.f}), 1.f},
            {new TouchBallReward(), 0.1f},
        },
        true
    );
    auto terminals = std::vector<TerminalCondition*>{
        new NoTouchCondition(10 * 120 / kFrameskip), new GoalScoreCondition(),
    };
    auto match = new Match(
        rewards,
        terminals,
        new DefaultOBS(),
        new DiscreteAction(),
        new KickoffState(),
        1,
        true
    );
    return {match, new Gym(match, kFrameskip)};
}

void ReportIteration(Learner*, Report& report) {
    if (reportedIterations++ < warmupIterations) return;
    const auto value = [&](const char* name) {
        return report.Has(name) ? report[name] : 0.0;
    };
    const double transitions       = value("Timesteps Collected");
    const double collectionSeconds = value("Collection Time");
    const double totalSeconds      = value("Total Iteration Time");
    logFile << "{\"kind\":\"native_iteration\",\"engine\":\"rlgymppo_cpp\""
            << ",\"profile\":\"unified\",\"threads\":" << threads
            << ",\"games_per_thread\":" << gamesPerThread
            << ",\"decision_environments\":" << threads * gamesPerThread * 2
            << ",\"frameskip\":" << kFrameskip
            << ",\"transitions\":" << transitions
            << ",\"collection_ticks\":" << transitions * kFrameskip
            << ",\"collection_seconds\":" << collectionSeconds
            << ",\"ppo_update_seconds\":" << totalSeconds - collectionSeconds
            << ",\"collection_ticks_per_second\":"
            << (collectionSeconds ? transitions * kFrameskip / collectionSeconds : 0)
            << ",\"total_ticks_per_second\":"
            << (totalSeconds ? transitions * kFrameskip / totalSeconds : 0)
            << "}\n";
    logFile.flush();
}
}

int main(int argc, char** argv) {
    int iterations = 1;
    std::filesystem::path output = "results/benchmarks.jsonl";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup-iterations" && i + 1 < argc) {
            warmupIterations = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--games-per-thread" && i + 1 < argc) {
            gamesPerThread = std::stoi(argv[++i]);
        } else if (arg == "--log-file" && i + 1 < argc) {
            output = argv[++i];
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--iterations N] [--warmup-iterations N] [--epochs N]"
                      << " [--threads N] [--games-per-thread N] [--log-file PATH]\n";
            return 2;
        }
    }
    if (
        iterations < 1 || warmupIterations < 0 || epochs < 1 || threads < 1
        || gamesPerThread < 1
    ) {
        return 2;
    }
    std::filesystem::create_directories(output.parent_path());
    logFile.open(output, std::ios::app);
    RocketSim::Init(COLLISION_MESHES_DIR);
    LearnerConfig config = {};
    config.numThreads            = threads;
    config.numGamesPerThread     = gamesPerThread;
    config.timestepsPerIteration = kCollection;
    config.timestepLimit         = static_cast<uint64_t>(iterations + warmupIterations)
                                 * kCollection;
    config.expBufferSize          = kCollection * 3;
    config.ppo.batchSize          = kCollection;
    config.ppo.miniBatchSize      = kMinibatch;
    config.ppo.epochs             = epochs;
    config.ppo.entCoef            = .01f;
    config.ppo.policyLR           = config.ppo.criticLR = 2e-4f;
    config.ppo.policyLayerSizes = {256, 256, 256};
    config.ppo.criticLayerSizes = {256, 256, 256};
    config.checkpointLoadFolder.clear();
    config.checkpointSaveFolder.clear();
    config.sendMetrics = false;
    Learner learner(CreateEnv, config);
    learner.iterationCallback = ReportIteration;
    learner.Learn();
}
