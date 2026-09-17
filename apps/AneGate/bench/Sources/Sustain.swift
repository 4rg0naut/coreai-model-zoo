// Sustain — sustained-generation + battery/thermal log for the ANE lane (S1, 2026-09-15).
// Self-driving: one launch runs a SEQUENCE of embedded bundles, each = idle → load → warmup →
// repeat the llm-benchmark trial (same splitmix64 prompt, greedy, AB_P/AB_G) until AB_DURATION_S
// elapsed. Every trial line carries wall clock, elapsed s, prompt/gen tok/s, generated tokens,
// battery level/state and thermal state, so the trajectory and the battery delta per token can
// be read from the log. Lines go to the console AND to Documents/sustain/<tag>.log in the shared
// container (survives a USB unplug — the battery-delta protocol needs the cable out).
//
// Env knobs (activate with AB_DURATION_S > 0):
//   AB_SEQUENCE   comma list of embedded bundle dir names, run in order (default: AB_MODEL / first)
//   AB_DURATION_S seconds of timed trials per arm (default 600)
//   AB_IDLE_S     minimum idle seconds before each arm, engine released (default 180)
//   AB_IDLE_TARGET idle continues until thermal state <= this (nominal|fair, default fair), AB_IDLE_MAX_S cap (600)
//   AB_P / AB_G   trial shape (default 128 / 256 = the same-day A/B shape)
//   AB_TAG        log file stem (default "sustain")
//
// Grammar:
//   SUSTAIN arm=<name> phase=idle|load|warmup|trial ... battery=<pct> bstate=<state> thermal=<state>
//   SUSTAIN_STATS arm=<name> trials=<n> gen_tokens=<n> gen_s=<s> gen_tps_first3=<x> gen_tps_last3=<x>
//                 gen_tps_min=<x> gen_tps_max=<x> battery_start=<pct> battery_end=<pct> thermal_max=<state>
//   SUSTAIN_DONE arms=<n>

import CoreAILanguageModels
import Foundation
import UIKit

enum Sustain {
    static var isRequested: Bool { Bench.envInt("AB_DURATION_S", 0) > 0 }

    @MainActor
    static func run(log: BenchLog) async {
        let durationS = Double(Bench.envInt("AB_DURATION_S", 600))
        let idleS = Double(Bench.envInt("AB_IDLE_S", 180))
        let p = Bench.envInt("AB_P", 128)
        let g = Bench.envInt("AB_G", 256)
        let tag = Bench.envStr("AB_TAG") ?? "sustain"
        let sequence: [String] = {
            if let s = Bench.envStr("AB_SEQUENCE") {
                return s.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
            }
            if let m = Bench.envStr("AB_MODEL") { return [m] }
            return Bench.findEmbeddedBundle().map { [$0.lastPathComponent] } ?? []
        }()

        UIApplication.shared.isIdleTimerDisabled = true   // screen stays on (lock cap trap)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let sink = LogSink(tag: tag)
        func emit(_ line: String) {
            sink.write(line)
            log.add(line)
        }
        emit("SUSTAIN_START tag=\(tag) sequence=\(sequence.joined(separator: ",")) duration_s=\(Int(durationS)) idle_s=\(Int(idleS)) p=\(p) g=\(g) free_gb=\(freeDiskGB()) wall=\(Date().ISO8601Format()) log=\(sink.url.path)")

        for arm in sequence {
            guard let modelDir = Bench.findEmbeddedBundle(named: arm) else {
                emit("FATAL arm=\(arm) not embedded")
                continue
            }
            // Idle with no engine alive so the chip cools: at least AB_IDLE_S, then keep waiting until
            // the thermal state is back at or below AB_IDLE_TARGET (nominal|fair, default fair) or
            // AB_IDLE_MAX_S (default 600) is reached — so every arm starts from a comparable state
            // whatever ran before it (the fixed 3 min let a GPU arm start at `serious` after an ANE arm).
            emit("SUSTAIN arm=\(arm) phase=idle_start \(state())")
            let idleStart = SuspendingClock.now
            if idleS > 0 { try? await Task.sleep(for: .seconds(idleS)) }
            let target: ProcessInfo.ThermalState = (Bench.envStr("AB_IDLE_TARGET") ?? "fair") == "nominal" ? .nominal : .fair
            let idleMaxS = Double(Bench.envInt("AB_IDLE_MAX_S", 600))
            while ProcessInfo.processInfo.thermalState.rawValue > target.rawValue,
                  Bench.seconds(from: idleStart, to: .now) < idleMaxS {
                try? await Task.sleep(for: .seconds(10))
            }
            emit("SUSTAIN arm=\(arm) phase=idle_end idle_s=\(fmt(Bench.seconds(from: idleStart, to: .now), 0)) target=\(thermalName(target)) \(state())")

            do {
                let bundle = try LanguageBundle(from: modelDir.path)
                let engineConfig = ModelConfig(
                    name: bundle.name,
                    tokenizer: bundle.tokenizer,
                    vocabSize: bundle.vocabSize,
                    maxContextLength: bundle.maxContextLength,
                    serializedModel: [bundle.modelAssetPath],
                    function: bundle.language.functionMap?.name(for: "main") ?? "main"
                )
                let configData = try JSONEncoder().encode(engineConfig)
                let loadStart = SuspendingClock.now
                let engine = try await EngineFactory.createEngine(
                    config: configData,
                    modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
                )
                let loadS = Bench.seconds(from: loadStart, to: .now)
                emit("SUSTAIN arm=\(arm) phase=load load_s=\(fmt(loadS, 3)) \(state())")

                let prompt = Bench.randomPrompt(vocabSize: bundle.vocabSize, count: p, seed: 0)
                let sampling = SamplingConfiguration(temperature: 0)
                let w = try await trial(engine: engine, prompt: prompt, sampling: sampling, g: g)
                emit("SUSTAIN arm=\(arm) phase=warmup prompt_tps=\(fmt(w.promptTps, 1)) gen_tps=\(fmt(w.genTps, 2)) tokens=\(w.tokens) \(state())")

                let batteryStart = batteryPct()
                var thermalMax = ProcessInfo.processInfo.thermalState
                var genTps: [Double] = []
                var promptTps: [Double] = []
                var tokens = 0
                var genSeconds = 0.0
                let t0 = SuspendingClock.now
                var i = 0
                while Bench.seconds(from: t0, to: .now) < durationS {
                    i += 1
                    let r = try await trial(engine: engine, prompt: prompt, sampling: sampling, g: g)
                    genTps.append(r.genTps)
                    promptTps.append(r.promptTps)
                    tokens += r.tokens
                    genSeconds += r.genSeconds
                    let ts = ProcessInfo.processInfo.thermalState
                    if ts.rawValue > thermalMax.rawValue { thermalMax = ts }
                    emit("SUSTAIN arm=\(arm) phase=trial i=\(i) elapsed_s=\(fmt(Bench.seconds(from: t0, to: .now), 1)) prompt_tps=\(fmt(r.promptTps, 1)) gen_tps=\(fmt(r.genTps, 2)) tokens=\(r.tokens) \(state())")
                }
                let batteryEnd = batteryPct()
                let first3 = genTps.prefix(3)
                let last3 = genTps.suffix(3)
                emit("SUSTAIN_STATS arm=\(arm) model=\(bundle.name) p=\(p) g=\(g) trials=\(genTps.count) gen_tokens=\(tokens) gen_s=\(fmt(genSeconds, 1)) elapsed_s=\(fmt(Bench.seconds(from: t0, to: .now), 1)) prompt_tps_mean=\(fmt(mean(promptTps), 1)) gen_tps_mean=\(fmt(mean(genTps), 2)) gen_tps_first3=\(fmt(mean(Array(first3)), 2)) gen_tps_last3=\(fmt(mean(Array(last3)), 2)) gen_tps_min=\(fmt(genTps.min() ?? 0, 2)) gen_tps_max=\(fmt(genTps.max() ?? 0, 2)) battery_start=\(batteryStart) battery_end=\(batteryEnd) bstate=\(batteryStateName()) thermal_max=\(thermalName(thermalMax)) load_s=\(fmt(loadS, 3)) footprint_gb=\(fmt(Double(Bench.currentPhysFootprint()) / 1_073_741_824.0, 2))")
                _ = engine   // released at scope exit
            } catch {
                emit("FATAL arm=\(arm) \(error)")
            }
        }
        emit("SUSTAIN_DONE arms=\(sequence.count) wall=\(Date().ISO8601Format())")
    }

    struct Trial {
        let promptTps: Double
        let genTps: Double
        let tokens: Int
        let genSeconds: Double
    }

    // BenchmarkMain.runTrial with the token count kept: prompt time = until the first stream
    // element, genTps over count-1 tokens after it.
    static func trial(
        engine: any InferenceEngine, prompt: [Int32], sampling: SamplingConfiguration, g: Int
    ) async throws -> Trial {
        try? await Task.sleep(for: .milliseconds(50))
        try await engine.reset()
        let options = InferenceOptions(maxTokens: g, includeLogits: false)
        let start = SuspendingClock.now
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: sampling, inferenceOptions: options)
        var promptTime = 0.0
        var genStart = SuspendingClock.now
        var count = 0
        for try await _ in stream {
            if promptTime == 0 {
                let now = SuspendingClock.now
                promptTime = Bench.seconds(from: start, to: now)
                genStart = now
            }
            count += 1
        }
        let genTime = Bench.seconds(from: genStart, to: .now)
        let decodeCount = max(0, count - 1)
        return Trial(
            promptTps: promptTime > 0 ? Double(prompt.count) / promptTime : 0,
            genTps: genTime > 0 ? Double(decodeCount) / genTime : 0,
            tokens: count, genSeconds: genTime)
    }

    @MainActor static func batteryPct() -> Int {
        let l = UIDevice.current.batteryLevel
        return l < 0 ? -1 : Int((l * 100).rounded())
    }

    @MainActor static func batteryStateName() -> String {
        switch UIDevice.current.batteryState {
        case .unplugged: return "unplugged"
        case .charging: return "charging"
        case .full: return "full"
        default: return "unknown"
        }
    }

    static func thermalName(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    @MainActor static func state() -> String {
        "battery=\(batteryPct()) bstate=\(batteryStateName()) thermal=\(thermalName(ProcessInfo.processInfo.thermalState)) wall=\(Date().ISO8601Format())"
    }

    // Free space on the data volume (what the specialization cache and installs draw from).
    static func freeDiskGB() -> String {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        if let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let cap = v.volumeAvailableCapacityForImportantUsage {
            return String(format: "%.1f", Double(cap) / 1_073_741_824.0)
        }
        return "?"
    }

    static func mean(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
    static func fmt(_ x: Double, _ d: Int) -> String { String(format: "%.\(d)f", x) }

    // Append-only log file in the shared container's Documents/sustain/ (not Documents/models).
    final class LogSink: @unchecked Sendable {
        let url: URL
        private let handle: FileHandle?
        init(tag: String) {
            let fm = FileManager.default
            let dir = fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("sustain")
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            url = dir.appendingPathComponent("\(tag).log")
            if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
            handle = try? FileHandle(forWritingTo: url)
            _ = try? handle?.seekToEnd()
        }
        func write(_ line: String) {
            guard let d = (line + "\n").data(using: .utf8) else { return }
            try? handle?.write(contentsOf: d)
            try? handle?.synchronize()
        }
    }
}
