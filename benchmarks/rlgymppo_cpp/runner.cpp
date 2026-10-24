#include <filesystem>
#include <fstream>
#include <iostream>

#include <RLGymPPO_CPP/Learner.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/DefaultOBS.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/StateSetters/KickoffState.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

using namespace RLGPC;
using namespace RLGSC;

namespace {

constexpr int kMinibatch         = 25000;
int           collectionTransitions = 100000;
int           frameskip          = 8;
int           threads            = 16;
int           gamesPerThread     = 24;
int           warmupIterations   = 1;
int           epochs             = 1;
int           reportedIterations = 0;
std::ofstream logFile;

EnvCreateResult CreateEnv() {
    auto rewards = new CombinedReward(
        {
            {new EventReward({.teamGoal = 10.f, .concede = -10.f}), 1.f},
            {new TouchBallReward(), 0.1f},
        },
        true
    );

    auto terminals = std::vector<TerminalCondition*>{
        new NoTouchCondition(10 * 120 / frameskip), new GoalScoreCondition(),
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

    return {match, new Gym(match, frameskip)};
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
            << ",\"frameskip\":" << frameskip
            << ",\"target_collection_transitions\":" << collectionTransitions
            << ",\"transitions\":" << transitions
            << ",\"collection_ticks\":" << transitions * frameskip
            << ",\"collection_seconds\":" << collectionSeconds
            << ",\"ppo_update_seconds\":" << totalSeconds - collectionSeconds
            << ",\"collection_ticks_per_second\":"
            << (collectionSeconds ? transitions * frameskip / collectionSeconds : 0)
            << ",\"total_ticks_per_second\":"
            << (totalSeconds ? transitions * frameskip / totalSeconds : 0)
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
        } else if (arg == "--collection-transitions" && i + 1 < argc) {
            collectionTransitions = std::stoi(argv[++i]);
        } else if (arg == "--warmup-iterations" && i + 1 < argc) {
            warmupIterations = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--games-per-thread" && i + 1 < argc) {
            gamesPerThread = std::stoi(argv[++i]);
        } else if (arg == "--frameskip" && i + 1 < argc) {
            frameskip = std::stoi(argv[++i]);
        } else if (arg == "--log-file" && i + 1 < argc) {
            output = argv[++i];
        } else {
            std::cerr << "usage: " << argv[0]
                       << " [--iterations N] [--collection-transitions N] [--warmup-iterations N] [--epochs N]"
                       << " [--threads N] [--games-per-thread N] [--frameskip N]"
                       << " [--log-file PATH]\n";
            return 2;
        }
    }
    if (
        iterations < 1 || collectionTransitions < 1 || warmupIterations < 0 || epochs < 1 || threads < 1
        || gamesPerThread < 1 || frameskip < 1
    ) {
        return 2;
    }

    std::filesystem::create_directories(output.parent_path());
    logFile.open(output, std::ios::app);
    RocketSim::Init(COLLISION_MESHES_DIR);
    LearnerConfig config = {};
    config.numThreads = threads;
    config.numGamesPerThread = gamesPerThread;
    config.timestepsPerIteration = collectionTransitions;
    config.timestepLimit = static_cast<uint64_t>(iterations + warmupIterations)
        * collectionTransitions;
    config.expBufferSize = collectionTransitions * 3;
    config.ppo.batchSize = collectionTransitions;
    config.ppo.miniBatchSize = kMinibatch;
    config.ppo.epochs = epochs;
    config.ppo.entCoef = .01f;
    config.ppo.policyLR = config.ppo.criticLR = 2e-4f;
    config.ppo.policyLayerSizes = {256, 256, 256};
    config.ppo.criticLayerSizes = {256, 256, 256};
    config.checkpointLoadFolder.clear();
    config.checkpointSaveFolder.clear();
    config.sendMetrics = false;
    Learner learner(CreateEnv, config);
    learner.iterationCallback = ReportIteration;
    learner.Learn();

}
