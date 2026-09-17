// AneGateRunner — on-device token-match gate for Apple's stock static (chunked-static,
// Neural Engine) iOS bundles, driven through the unmodified Apple main package
// (vendor/coreai-models-main-7359dbc, git archive) via EngineFactory -> StaticShapeEngine.
//
// Judges the embedded bundle against an fp32 HF oracle fixture (fixtures/*.json, made by
// make_fixture.py) with the zoo's existing rules (cli/coreai_verify.py, knowledge/pipelined-engine.md):
//   1. TF  teacher-forced single-step sweep: the oracle's continuation is fed as
//          `forcedContinuation`, the engine's own logits at every step are argmax'd and
//          compared with the oracle's token at that step. A mismatch where the oracle's
//          top-2 margin clears the floor is a FAIL; below the floor it is a knife-edge tie
//          (reported, excluded).
//   2. FR  free-running greedy rollout (temperature 0) from the same prompt: judged at the
//          first diverging step by the oracle's margin there. The STOP is a token too: an
//          oracle EOS on a margin-clear step that the engine does not emit is a FAIL, and
//          an engine EOS where the oracle continues is a divergence like any other.
// PASS needs both. One line per prompt, `GATE ...`, then `GATE_SUMMARY ... VERDICT=PASS|FAIL`.
//
// Env knobs: AG_FIXTURE (fixture file stem in fixtures/, default "fixture"),
//            AG_PROMPTS (comma list of prompt names, default all),
//            AG_EXTRA (free-run cap = expected + AG_EXTRA, default 8),
//            AG_MODEL (S5: name of the embedded bundle dir to gate when several are embedded — one
//                      install per arm set instead of one per arm; default = the first found).
// Runs on launch; results on screen and on the console (devicectl --console).

import CoreAILanguageModels
import Darwin
import Foundation
import SwiftUI

@main
struct AneGateRunnerApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class GateLog: ObservableObject {
    @Published var lines: [String] = []
    @Published var running = false

    // S1: mirror to Documents/sustain/<AG_LOG>.log (no-console launches).
    private let sink = GateFileSink(tag: ProcessInfo.processInfo.environment["AG_LOG"] ?? "gate")
    func add(_ line: String) {
        print(line)
        fflush(stdout)
        sink.write(line)
        lines.append(line)
    }
}

struct ContentView: View {
    @StateObject private var log = GateLog()

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(log.lines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
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
                Task { await runGate() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(log.running)
            .padding(.bottom, 8)
        }
        .task { await runGate() }
    }

    @MainActor
    private func runGate() async {
        guard !log.running else { return }
        log.running = true
        await Gate.run(log: log)
        log.running = false
    }
}

// MARK: - Fixture

struct FixturePrompt: Decodable {
    let name: String
    let prompt_text: String
    let chat: String?
    let prompt_ids: [Int32]
    let expected_ids: [Int32]
    let expected_text: String
    let margins: [Double]         // fp32 oracle top-2 softmax-probability gap per step
    let margins_logit: [Double]   // fp32 oracle top-2 raw-logit gap per step (record only)
    let stopped_on_eos: Bool
    let eos_ids: [Int32]
    let must_stop_within: Int?
}

struct Fixture: Decodable {
    struct Poison: Decodable {
        let prompt: String
        let step: Int
        let was: Int32
        let now: Int32
    }
    let hf_id: String
    let dtype: String
    let floor: Double
    let poison: Poison?
    let prompts: [FixturePrompt]
}

// MARK: - Gate

enum Gate {
    static func envInt(_ name: String, _ fallback: Int) -> Int {
        guard let raw = getenv(name), let v = Int(String(cString: raw)) else { return fallback }
        return v
    }

    static func envStr(_ name: String) -> String? {
        guard let raw = getenv(name) else { return nil }
        let s = String(cString: raw)
        return s.isEmpty ? nil : s
    }

    static func run(log: GateLog) async {
        let fixtureStem = envStr("AG_FIXTURE") ?? "fixture"
        let extra = envInt("AG_EXTRA", 8)
        let onlyPrompts = Set((envStr("AG_PROMPTS") ?? "").split(separator: ",").map(String.init))

        do {
            guard let modelDir = findEmbeddedBundle(named: envStr("AG_MODEL")) else {
                await log.add("FATAL no embedded model bundle (dir with metadata.json) found (AG_MODEL=\(envStr("AG_MODEL") ?? "-"))")
                return
            }
            guard let fixtureURL = Bundle.main.resourceURL?
                .appendingPathComponent("fixtures/\(fixtureStem).json"),
                FileManager.default.fileExists(atPath: fixtureURL.path)
            else {
                await log.add("FATAL fixture fixtures/\(fixtureStem).json not in app resources")
                return
            }
            let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
            await log.add("model dir: \(modelDir.lastPathComponent)")
            await log.add(
                "fixture: \(fixtureStem).json oracle=\(fixture.hf_id) \(fixture.dtype) floor=\(fixture.floor) "
                    + "prompts=\(fixture.prompts.map(\.name).joined(separator: ","))")
            if let p = fixture.poison {
                await log.add("fixture is POISONED: \(p.prompt)[\(p.step)] \(p.was) -> \(p.now) (must FAIL)")
            }

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

            await log.add("loading engine...")
            let loadStart = SuspendingClock.now
            let engine = try await EngineFactory.createEngine(
                config: configData,
                modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
            )
            let loadSeconds = seconds(from: loadStart, to: .now)
            await log.add(String(format: "engine loaded in %.3fs supportsLogits=%d", loadSeconds,
                                 engine.supportsLogits ? 1 : 0))
            guard engine.supportsLogits else {
                await log.add("FATAL engine does not expose logits; the TF sweep needs them")
                return
            }

            var passCount = 0
            var failCount = 0
            for prompt in fixture.prompts {
                if !onlyPrompts.isEmpty && !onlyPrompts.contains(prompt.name) { continue }
                let verdict = try await gatePrompt(
                    engine: engine, prompt: prompt, floor: fixture.floor, extra: extra, log: log)
                if verdict { passCount += 1 } else { failCount += 1 }
            }
            let footprintGB = Double(currentPhysFootprint()) / 1_073_741_824.0
            await log.add(String(
                format: "GATE_SUMMARY fixture=%@ model=%@ prompts=%d pass=%d fail=%d load_s=%.3f footprint_gb=%.2f VERDICT=%@",
                fixtureStem, bundle.name, passCount + failCount, passCount, failCount, loadSeconds,
                footprintGB, failCount == 0 && passCount > 0 ? "PASS" : "FAIL"))
        } catch {
            await log.add("FATAL \(error)")
        }
    }

    /// Runs the TF sweep and the FR rollout for one prompt; returns true on PASS.
    static func gatePrompt(
        engine: any InferenceEngine, prompt: FixturePrompt, floor: Double, extra: Int, log: GateLog
    ) async throws -> Bool {
        let expected = prompt.expected_ids
        let n = expected.count
        let eos = Set(prompt.eos_ids)
        let greedy = SamplingConfiguration(temperature: 0)
        await log.add("---- prompt \(prompt.name): \(prompt.prompt_ids.count) ids, expected \(n) steps, "
                      + "oracle_stop=\(prompt.stopped_on_eos) text=\(prompt.expected_text.debugDescription)")

        // 1. Teacher-forced sweep.
        try await engine.reset(to: 0)
        let tfOptions = InferenceOptions(maxTokens: nil, includeLogits: true, forcedContinuation: expected)
        let tfStart = SuspendingClock.now
        let tfStream = try await engine.generate(
            with: prompt.prompt_ids, samplingConfiguration: greedy, inferenceOptions: tfOptions)
        var tfSteps = 0
        var tfOK = 0
        var tfKnife = 0
        var tfFail = 0
        var tfGot: [Int32] = []
        for try await out in tfStream {
            let k = tfSteps
            guard let logits = out.logits, k < n else {
                await log.add("TF \(prompt.name) k=\(k) FATAL no logits in output")
                tfFail += 1
                break
            }
            let (top1, top2, gap) = top2(logits)
            tfGot.append(top1)
            let m = prompt.margins[k]
            let status: String
            if top1 == expected[k] {
                status = "ok"
                tfOK += 1
            } else if m < floor {
                status = "KNIFE"
                tfKnife += 1
            } else {
                status = "FAIL"
                tfFail += 1
            }
            await log.add(String(
                format: "TF %@ k=%d exp=%d got=%d second=%d oracle_margin=%.4f dev_gap=%.3f %@",
                prompt.name, k, expected[k], top1, top2, m, gap, status))
            tfSteps += 1
        }
        let tfSeconds = seconds(from: tfStart, to: .now)
        if tfSteps < n {
            await log.add("TF \(prompt.name) FAIL stream ended after \(tfSteps)/\(n) steps")
            tfFail += 1
        }
        let tfPass = tfFail == 0

        // 2. Free-running greedy rollout.
        try await engine.reset(to: 0)
        let cap = n + extra
        let frOptions = InferenceOptions(maxTokens: cap, includeLogits: false)
        let frStart = SuspendingClock.now
        let frStream = try await engine.generate(
            with: prompt.prompt_ids, samplingConfiguration: greedy, inferenceOptions: frOptions)
        var got: [Int32] = []
        for try await out in frStream {
            got.append(out.tokenId)
            if eos.contains(out.tokenId) { break }
        }
        let frSeconds = seconds(from: frStart, to: .now)
        let (frPass, frReason) = judgeRollout(
            got: got, prompt: prompt, floor: floor, cap: cap, eos: eos)
        await log.add("FR \(prompt.name) got(\(got.count))=\(got)")
        await log.add("FR \(prompt.name) \(frPass ? "PASS" : "FAIL") \(frReason)")

        let verdict = tfPass && frPass
        await log.add(String(
            format: "GATE prompt=%@ tf=%d/%d knife=%d fail=%d fr=%@ tf_s=%.2f fr_s=%.2f verdict=%@",
            prompt.name, tfOK, n, tfKnife, tfFail, frPass ? "PASS" : "FAIL", tfSeconds, frSeconds,
            verdict ? "PASS" : "FAIL"))
        return verdict
    }

    /// coreai_verify.judge at token level: the first diverging step is judged by the
    /// oracle's margin there; the stop is a step like any other.
    static func judgeRollout(
        got: [Int32], prompt: FixturePrompt, floor: Double, cap: Int, eos: Set<Int32>
    ) -> (Bool, String) {
        let expected = prompt.expected_ids
        let margins = prompt.margins
        let n = expected.count
        let common = min(got.count, n)
        var d = -1
        for i in 0..<common where got[i] != expected[i] {
            d = i
            break
        }
        let devStopped = got.last.map { eos.contains($0) } ?? false

        if d < 0 && got.count < n {
            // No disagreement but the engine produced fewer tokens than the oracle without
            // an EOS: the stream ended early (error/cancel), which is not a judgement.
            return (false, "stream ended after \(got.count)/\(n) tokens with no EOS")
        }
        if d < 0 {
            // Every oracle step matched, stop included when the oracle stopped (its EOS is
            // the last expected token, so a matching got[n-1] IS the engine's stop).
            if prompt.stopped_on_eos {
                if let w = prompt.must_stop_within, got.count > w {
                    return (false, "\(n)/\(n) exact but EOS came after must_stop_within=\(w)")
                }
                return (true, String(format: "%d/%d token-exact incl. the stop (EOS at step %d, oracle margin %.3f)",
                                     n, n, n - 1, margins[n - 1]))
            }
            if devStopped && got.count <= n {
                // Cannot happen: an engine EOS inside the oracle's span would have mismatched.
                return (false, "engine stopped inside the oracle span without a mismatch (unexpected)")
            }
            return (true, "\(n)/\(n) token-exact; engine continued past the oracle's \(n)-token span to \(got.count) (nothing disagrees)")
        }
        // Divergence at step d.
        let m = margins[d]
        let what: String
        if eos.contains(got[d]) {
            what = "engine stopped (EOS) where the oracle continues"
        } else if eos.contains(expected[d]) {
            what = "oracle stops (EOS) and the engine kept generating"
        } else {
            what = "token disagreement"
        }
        if m >= floor {
            return (false, String(format: "%d/%d exact, then %@ at step %d: exp=%d got=%d, oracle margin %.4f >= %.2f floor",
                                  d, n, what, d, expected[d], got[d], m, floor))
        }
        if let w = prompt.must_stop_within, !got.contains(where: { eos.contains($0) }) || got.count > w {
            return (false, String(format: "knife-edge fork at step %d (margin %.4f) but no EOS within must_stop_within=%d",
                                  d, m, w))
        }
        return (true, String(format: "%d/%d exact, then %@ at step %d on a %.4f margin (< %.2f floor): knife-edge, not a defect; rollout beyond it is not gated (the TF sweep covers those steps)",
                             d, n, what, d, m, floor))
    }

    /// Top-2 of a Float16 logits row (computed in Float): (argmax, second, gap).
    static func top2(_ logits: [LogitsScalarType]) -> (Int32, Int32, Float) {
        var best = -Float.infinity
        var bestIdx = 0
        var second = -Float.infinity
        var secondIdx = 0
        for (i, v) in logits.enumerated() {
            let f = Float(v)
            if f > best {
                second = best
                secondIdx = bestIdx
                best = f
                bestIdx = i
            } else if f > second {
                second = f
                secondIdx = i
            }
        }
        return (Int32(bestIdx), Int32(secondIdx), best - second)
    }

    static func seconds(from start: SuspendingClock.Instant, to end: SuspendingClock.Instant) -> Double {
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    // First resource subdirectory containing metadata.json = the model bundle (or the one named by AG_MODEL).
    static func findEmbeddedBundle(named: String? = nil) -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else { return nil }
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: resourceURL, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }
        for entry in entries {
            if (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                fm.fileExists(atPath: entry.appendingPathComponent("metadata.json").path) {
                if let named, entry.lastPathComponent != named { continue }
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

// Append-only log file in the shared container's Documents/sustain/ (S1, 2026-09-15).
final class GateFileSink: @unchecked Sendable {
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
