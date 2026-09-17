// AppleBenchRunner — iPhone port of Apple's official llm-benchmark
// (coreai-models/swift/Sources/Tools/benchmark/BenchmarkMain.swift).
// Same methodology as the Mac numbers: synthetic random prompt (identical
// splitmix64 generator), greedy sampling, 1 warmup trial + N timed trials,
// genTps excludes the first emitted token. The embedded bundle folder
// (project.yml) selects the model under test.
//
// Runs automatically on launch; results on screen and in the console
// (devicectl --console, grep "STATS|FATAL").
// Env knobs: AB_P / AB_G / AB_N (default 512/1024/5 = llm-benchmark defaults);
// AB_MODEL = name of the embedded bundle dir to load when more than one is embedded (A/B runs).

import CoreAILanguageModels
import Darwin
import Foundation
import SwiftUI

@main
struct AppleBenchRunnerApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class BenchLog: ObservableObject {
    @Published var lines: [String] = []
    @Published var running = false

    // S1: every line also goes to Documents/sustain/<AB_LOG>.log so a no-console launch can be read back
    // (over some transports `devicectl … launch --console` is refused with CoreDeviceError 10002).
    private let sink = Sustain.LogSink(tag: Bench.envStr("AB_LOG") ?? "bench")
    func add(_ line: String) {
        print(line)
        fflush(stdout)
        sink.write(line)
        lines.append(line)
    }
}

struct ContentView: View {
    @StateObject private var log = BenchLog()

    var body: some View {
        if Demo.isRequested {   // S1: AB_DEMO=1 = recordable chat demo (Demo.swift)
            DemoView(log: log)
        } else {
            benchBody
        }
    }

    private var benchBody: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(log.lines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 12, design: .monospaced))
                            .textSelection(.enabled)
                    }
                    Color.clear.frame(height: 1).id("end")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            .onChange(of: log.lines.count) {
                proxy.scrollTo("end", anchor: .bottom)
            }
        }
        .safeAreaInset(edge: .bottom) {
            Button(log.running ? "Running..." : "Re-run") {
                Task { await runBench() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(log.running)
            .padding(.bottom, 8)
        }
        .task { await runBench() }
    }

    @MainActor
    private func runBench() async {
        guard !log.running else { return }
        log.running = true
        await Bench.run(log: log)
        log.running = false
    }
}

enum Bench {
    static func envStr(_ name: String) -> String? {
        guard let raw = getenv(name) else { return nil }
        let s = String(cString: raw)
        return s.isEmpty ? nil : s
    }

    static func envInt(_ name: String, _ fallback: Int) -> Int {
        guard let raw = getenv(name), let v = Int(String(cString: raw)) else { return fallback }
        return v
    }

    static func run(log: BenchLog) async {
        if Sustain.isRequested { await Sustain.run(log: log); return }   // S1: AB_DURATION_S>0 = sustained/battery mode (Sustain.swift)
        if TaskEval.isRequested { await TaskEval.run(log: log); return }   // S5: AB_TASK=<stem> = task-accuracy run (TaskEval.swift)
        let promptTokens = envInt("AB_P", 512)
        let generationTokens = envInt("AB_G", 1024)
        let numTrials = envInt("AB_N", 5)

        do {
            guard let modelDir = findEmbeddedBundle(named: envStr("AB_MODEL")) else {
                await log.add("FATAL no embedded model bundle (dir with metadata.json) found (AB_MODEL=\(envStr("AB_MODEL") ?? "-"))")
                return
            }
            await log.add("model dir: \(modelDir.lastPathComponent) free_gb=\(Sustain.freeDiskGB())")

            let bundle = try LanguageBundle(from: modelDir.path)
            let vocabSize = bundle.vocabSize

            let engineConfig = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            let configData = try JSONEncoder().encode(engineConfig)

            await log.add("loading engine...")
            let loadStart = SuspendingClock.now
            let engine = try await EngineFactory.createEngine(
                config: configData,
                modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
            )
            let loadSeconds = seconds(from: loadStart, to: .now)
            await log.add(String(format: "engine loaded in %.3fs", loadSeconds))

            let prompt = randomPrompt(vocabSize: vocabSize, count: promptTokens, seed: 0)
            let sampling = SamplingConfiguration(temperature: 0)

            await log.add("warmup trial...")
            _ = try await runTrial(
                engine: engine, prompt: prompt, sampling: sampling,
                generationTokens: generationTokens)

            await log.add("benchmarking p=\(promptTokens) g=\(generationTokens) n=\(numTrials)")
            var promptTpsAll: [Double] = []
            var genTpsAll: [Double] = []
            for i in 0..<numTrials {
                let r = try await runTrial(
                    engine: engine, prompt: prompt, sampling: sampling,
                    generationTokens: generationTokens)
                promptTpsAll.append(r.promptTps)
                genTpsAll.append(r.genTps)
                await log.add(String(
                    format: "trial %d: prompt %.1f tok/s, gen %.2f tok/s", i + 1, r.promptTps, r.genTps))
            }

            let n = Double(numTrials)
            let avgPrompt = promptTpsAll.reduce(0, +) / n
            let avgGen = genTpsAll.reduce(0, +) / n
            let footprintGB = Double(currentPhysFootprint()) / 1_073_741_824.0
            await log.add(String(
                format: "STATS model=%@ p=%d g=%d n=%d prompt_tps=%.1f gen_tps=%.2f load_s=%.3f footprint_gb=%.2f",
                bundle.name, promptTokens, generationTokens, numTrials,
                avgPrompt, avgGen, loadSeconds, footprintGB))
        } catch {
            await log.add("FATAL \(error)")
        }
    }

    struct TrialResult {
        let promptTps: Double
        let genTps: Double
    }

    // Identical to BenchmarkMain.runTrial: prompt time = until first stream
    // element, genTps over count-1 tokens after it.
    static func runTrial(
        engine: any InferenceEngine,
        prompt: [Int32],
        sampling: SamplingConfiguration,
        generationTokens: Int
    ) async throws -> TrialResult {
        try? await Task.sleep(for: .milliseconds(50))
        try await engine.reset()

        let options = InferenceOptions(maxTokens: generationTokens, includeLogits: false)
        let start = SuspendingClock.now
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: sampling, inferenceOptions: options
        )

        var promptTime: Double = 0
        var genStart = SuspendingClock.now
        var count = 0

        for try await _ in stream {
            if promptTime == 0 {
                let now = SuspendingClock.now
                promptTime = seconds(from: start, to: now)
                genStart = now
            }
            count += 1
        }

        let genTime = seconds(from: genStart, to: .now)
        let promptTps = promptTime > 0 ? Double(prompt.count) / promptTime : 0
        let decodeCount = max(0, count - 1)
        let genTps = genTime > 0 ? Double(decodeCount) / genTime : 0
        return TrialResult(promptTps: promptTps, genTps: genTps)
    }

    // Identical splitmix64 synthetic prompt generator from BenchmarkMain.
    static func randomPrompt(vocabSize: Int, count: Int, seed: UInt64) -> [Int32] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var out = [Int32]()
        out.reserveCapacity(count)
        let v = UInt64(vocabSize)
        for _ in 0..<count {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            out.append(Int32(z % v))
        }
        return out
    }

    static func seconds(from start: SuspendingClock.Instant, to end: SuspendingClock.Instant) -> Double {
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    // Resource subdirectory containing metadata.json = a model bundle; `named` picks one when
    // several are embedded (A/B), otherwise the first found.
    static func findEmbeddedBundle(named: String? = nil) -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: resourceURL, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }
        for entry in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                fm.fileExists(atPath: entry.appendingPathComponent("metadata.json").path),
                named == nil || entry.lastPathComponent == named {
                return entry
            }
        }
        return nil
    }

    static func currentPhysFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
