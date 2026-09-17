// Demo — a screen-recordable run for the ANE lane (S1, 2026-09-16): one real chat question, answered
// by each embedded bundle in turn (e.g. the Neural Engine bundle, then the GPU bundle), the answer
// streamed as text with a live decode tok/s counter. Same engine path as the bench (EngineFactory →
// StaticShapeEngine / pipelined engine), greedy, the bundle's own tokenizer and chat template.
//
// Env knobs (activate with AB_DEMO=1):
//   AB_SEQUENCE   comma list of embedded bundle dir names (default: all embedded, sorted)
//   AB_QUESTION   the user turn (default below)
//   AB_MAX_TOKENS generation cap (default 220)
//   AB_WARMUP=1   run a hidden 8-token generation first so the visible run starts warm
//   AB_LABEL_<name> optional display label per bundle (default: the bundle's metadata name)
// Console/file grammar: DEMO arm=<name> phase=load|prompt|token|done ... ; DEMO_DONE arms=<n>

import CoreAILanguageModels
import Foundation
import SwiftUI
import Tokenizers

@MainActor
final class DemoState: ObservableObject {
    @Published var title = ""
    @Published var subtitle = ""
    @Published var status = "loading…"
    @Published var answer = ""
    @Published var tps = 0.0
    @Published var tokens = 0
    @Published var done = false
}

struct DemoView: View {
    @StateObject private var state = DemoState()
    let log: BenchLog

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(alignment: .leading, spacing: 14) {
                Text(state.title)
                    .font(.system(size: 26, weight: .bold))
                    .foregroundStyle(.white)
                Text(state.subtitle)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.gray)
                Divider().overlay(Color.gray)
                Text("Q: \(Demo.question)")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Color(red: 0.6, green: 0.85, blue: 1.0))
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(state.answer.isEmpty ? " " : state.answer)
                            .font(.system(size: 21))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Color.clear.frame(height: 1).id("end")
                    }
                    .onChange(of: state.answer.count) { proxy.scrollTo("end", anchor: .bottom) }
                }
                Divider().overlay(Color.gray)
                HStack(alignment: .lastTextBaseline) {
                    Text(state.tps > 0 ? String(format: "%.1f", state.tps) : "–")
                        .font(.system(size: 56, weight: .bold, design: .rounded))
                        .foregroundStyle(Color(red: 0.4, green: 1.0, blue: 0.6))
                        .monospacedDigit()
                    Text("tok/s decode").font(.system(size: 18)).foregroundStyle(.gray)
                    Spacer()
                    Text("\(state.tokens) tokens").font(.system(size: 18)).foregroundStyle(.gray).monospacedDigit()
                }
                Text(state.status).font(.system(size: 14, design: .monospaced)).foregroundStyle(.gray)
            }
            .padding(22)
        }
        .task { await Demo.run(state: state, log: log) }
    }
}

enum Demo {
    static var isRequested: Bool { Bench.envInt("AB_DEMO", 0) > 0 }
    static var question: String {
        Bench.envStr("AB_QUESTION") ?? "Explain in a short paragraph why the sky is blue."
    }

    @MainActor
    static func run(state: DemoState, log: BenchLog) async {
        UIApplication.shared.isIdleTimerDisabled = true
        let maxTokens = Bench.envInt("AB_MAX_TOKENS", 220)
        let sequence: [String] = {
            if let s = Bench.envStr("AB_SEQUENCE") {
                return s.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
            }
            guard let resourceURL = Bundle.main.resourceURL,
                  let entries = try? FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil) else { return [] }
            return entries.filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("metadata.json").path) }
                .map(\.lastPathComponent).sorted()
        }()
        log.add("DEMO_START sequence=\(sequence.joined(separator: ",")) question=\(question)")

        for arm in sequence {
            guard let modelDir = Bench.findEmbeddedBundle(named: arm) else { log.add("FATAL arm=\(arm) not embedded"); continue }
            state.answer = ""; state.tps = 0; state.tokens = 0
            do {
                let bundle = try LanguageBundle(from: modelDir.path)
                state.title = Bench.envStr("AB_LABEL_\(arm)") ?? bundle.name
                state.subtitle = "iPhone 17 Pro · iOS 27.0 · Core AI"
                state.status = "loading engine…"
                let engineConfig = ModelConfig(
                    name: bundle.name, tokenizer: bundle.tokenizer, vocabSize: bundle.vocabSize,
                    maxContextLength: bundle.maxContextLength, serializedModel: [bundle.modelAssetPath],
                    function: bundle.language.functionMap?.name(for: "main") ?? "main")
                let configData = try JSONEncoder().encode(engineConfig)
                let loadStart = SuspendingClock.now
                let engine = try await EngineFactory.createEngine(
                    config: configData, modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main))
                let loadS = Bench.seconds(from: loadStart, to: .now)
                log.add("DEMO arm=\(arm) phase=load load_s=\(String(format: "%.2f", loadS))")
                state.status = String(format: "engine loaded in %.2f s", loadS)

                guard let tokDir = bundle.tokenizerPath else { throw NSError(domain: "demo", code: 1, userInfo: [NSLocalizedDescriptionKey: "no embedded tokenizer"]) }
                let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
                let ids: [Int]
                do {
                    ids = try tokenizer.applyChatTemplate(
                        messages: [["role": "user", "content": question]], tools: nil,
                        additionalContext: ["enable_thinking": false])
                } catch {
                    log.add("DEMO arm=\(arm) chat template failed (\(error)); using the manual no-think form")
                    let manual = "<s><|im_start|>user\n\(question)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    ids = tokenizer.encode(text: manual, addSpecialTokens: false)
                }
                log.add("DEMO arm=\(arm) phase=prompt ids=\(ids)")

                if Bench.envInt("AB_WARMUP", 0) > 0 {   // optional: one short hidden generation so the visible run is warm (as the bench's warmup trial)
                    state.status = "warming up…"
                    let w = try await engine.generate(with: ids.map { Int32($0) }, samplingConfiguration: SamplingConfiguration(temperature: 0),
                                                      inferenceOptions: InferenceOptions(maxTokens: 8, includeLogits: false))
                    for try await _ in w {}
                    try await engine.reset()
                    log.add("DEMO arm=\(arm) phase=warmup done")
                }
                let options = InferenceOptions(maxTokens: maxTokens, includeLogits: false)
                let stream = try await engine.generate(
                    with: ids.map { Int32($0) }, samplingConfiguration: SamplingConfiguration(temperature: 0),
                    inferenceOptions: options)
                var generated: [Int] = []
                var firstAt: SuspendingClock.Instant? = nil
                let start = SuspendingClock.now
                state.status = "generating…"
                let eosId = tokenizer.eosTokenId   // <|im_end|> for the MiniCPM5 chat bundles (tokenizer_config eos)
                var stoppedOnEos = false
                for try await out in stream {
                    let now = SuspendingClock.now
                    if firstAt == nil { firstAt = now }
                    if let e = eosId, Int(out.tokenId) == e { stoppedOnEos = true; break }   // the bench engine path does not stop by itself
                    generated.append(Int(out.tokenId))
                    state.tokens = generated.count
                    state.answer = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                    if let f = firstAt, generated.count > 1 {
                        state.tps = Double(generated.count - 1) / Bench.seconds(from: f, to: now)
                    }
                    if generated.count == 1 || generated.count % 10 == 0 {
                        log.add("DEMO arm=\(arm) phase=token n=\(generated.count) t_s=\(String(format: "%.1f", Bench.seconds(from: start, to: now))) tps=\(String(format: "%.2f", state.tps))")
                    }
                }
                let total = Bench.seconds(from: start, to: .now)
                let ttft = firstAt.map { Bench.seconds(from: start, to: $0) } ?? 0
                state.status = String(format: "done · %d tokens · %.1f tok/s decode · first token %.2f s", generated.count, state.tps, ttft)
                log.add("DEMO arm=\(arm) phase=done eos=\(stoppedOnEos) tokens=\(generated.count) gen_tps=\(String(format: "%.2f", state.tps)) ttft_s=\(String(format: "%.2f", ttft)) total_s=\(String(format: "%.1f", total)) text=\(state.answer.replacingOccurrences(of: "\n", with: "\\n"))")
                _ = engine
            } catch {
                log.add("FATAL arm=\(arm) \(error)")
                state.status = "error: \(error)"
            }
            try? await Task.sleep(for: .seconds(4))
        }
        state.done = true
        log.add("DEMO_DONE arms=\(sequence.count)")
    }
}
