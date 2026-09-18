// TaskEval — task-accuracy runs on the phone (S5, 2026-09-16): every question of an embedded JSONL is asked through
// the bundle's own chat template (no-think), greedy, EOS-stopped; the raw answer text goes to a JSONL in the shared
// container so the Mac scores it with the same extraction/normalization as ~/code/litertlm-convert's GSM8K evals
// (`score_gsm8k.py`). One engine for the whole run, reset between questions. Same engine path as the bench/demo
// (EngineFactory → StaticShapeEngine for the ANE bundles, the pipelined engine for the GPU bundle).
//
// Env knobs (activate with AB_TASK=<stem of tasks/<stem>.jsonl>):
//   AB_MODEL       embedded bundle dir name (as the bench)
//   AB_MAX_TOKENS  generation cap per question (default 640: an iOS dynamic bundle truncates at position 1024)
//   AB_TASK_START / AB_TASK_N   item range (resume)
//   AB_SUFFIX      appended to every question (default: the litertlm-convert CoT suffix, see COT below)
//   AB_PROMPT_FORMAT  explicit single-turn template with {q} for the user text, encoded with no extra special
//                  tokens (S6: LFM2.5's Jinja template is beyond swift-transformers; the built-in fallback is
//                  MiniCPM/Qwen-shaped). LFM2.5: "<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
// Grammar: TASK_START …; TASK i=<n> tokens=<t> ttft_s=… gen_tps=… eos=<0|1> capped=<0|1>; TASK_DONE n=<n> …
// Answers: Documents/sustain/<AB_LOG>_answers.log, one JSON object per line {i, tokens, eos, capped, ttft_s, gen_s, text}.

import CoreAILanguageModels
import Foundation
import Tokenizers

enum TaskEval {
    static var isRequested: Bool { Bench.envStr("AB_TASK") != nil }
    static let COT = "\n\nSolve this step by step. After your reasoning, write the final answer on its own line in the exact form:\n#### <number>"

    struct Item: Decodable { let i: Int; let question: String; let gold: String }

    @MainActor
    static func run(log: BenchLog) async {
        let stem = Bench.envStr("AB_TASK") ?? "gsm8k_200"
        let maxTokens = Bench.envInt("AB_MAX_TOKENS", 640)
        let start = Bench.envInt("AB_TASK_START", 0)
        let limit = Bench.envInt("AB_TASK_N", 100_000)
        let suffix = Bench.envStr("AB_SUFFIX") ?? COT
        let answers = Sustain.LogSink(tag: (Bench.envStr("AB_LOG") ?? "task") + "_answers")
        do {
            guard let modelDir = Bench.findEmbeddedBundle(named: Bench.envStr("AB_MODEL")) else {
                log.add("FATAL no embedded model bundle (AB_MODEL=\(Bench.envStr("AB_MODEL") ?? "-"))"); return
            }
            guard let url = Bundle.main.resourceURL?.appendingPathComponent("tasks/\(stem).jsonl"),
                  let raw = try? String(contentsOf: url, encoding: .utf8) else {
                log.add("FATAL tasks/\(stem).jsonl not in app resources"); return
            }
            let items = try raw.split(separator: "\n").filter { !$0.isEmpty }
                .map { try JSONDecoder().decode(Item.self, from: Data($0.utf8)) }
            let todo = Array(items.dropFirst(start).prefix(limit))
            log.add("model dir: \(modelDir.lastPathComponent) free_gb=\(Sustain.freeDiskGB())")
            let bundle = try LanguageBundle(from: modelDir.path)
            let engineConfig = ModelConfig(
                name: bundle.name, tokenizer: bundle.tokenizer, vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength, serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main")
            let configData = try JSONEncoder().encode(engineConfig)
            let loadStart = SuspendingClock.now
            var engine = try await EngineFactory.createEngine(
                config: configData, modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main))
            let loadS = Bench.seconds(from: loadStart, to: .now)
            guard let tokDir = bundle.tokenizerPath else { log.add("FATAL no embedded tokenizer"); return }
            let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
            let eosId = tokenizer.eosTokenId
            log.add("TASK_START task=\(stem) model=\(bundle.name) items=\(todo.count) start=\(start) max_tokens=\(maxTokens) load_s=\(String(format: "%.2f", loadS)) eos=\(eosId.map(String.init) ?? "nil")")

            var done = 0
            let runStart = SuspendingClock.now
            for item in todo {
              do {   // per-question guard (S5 2026-09-17: one Metal "GPU Timeout" killed a whole int8 run at item 30) — log, recreate the engine, continue
                let ids: [Int]
                if let fmt = Bench.envStr("AB_PROMPT_FORMAT") {
                    ids = tokenizer.encode(text: fmt.replacingOccurrences(of: "{q}", with: item.question + suffix), addSpecialTokens: false)
                } else {
                do {
                    ids = try tokenizer.applyChatTemplate(
                        messages: [["role": "user", "content": item.question + suffix]], tools: nil,
                        additionalContext: ["enable_thinking": false])
                } catch {
                    let manual = "<s><|im_start|>user\n\(item.question + suffix)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    ids = tokenizer.encode(text: manual, addSpecialTokens: false)
                }
                }
                if done == 0 { log.add("prompt ids[0..<12]=\(Array(ids.prefix(12))) n=\(ids.count)") }
                try await engine.reset()
                let stream = try await engine.generate(
                    with: ids.map { Int32($0) }, samplingConfiguration: SamplingConfiguration(temperature: 0),
                    inferenceOptions: InferenceOptions(maxTokens: maxTokens, includeLogits: false))
                var generated: [Int] = []
                var firstAt: SuspendingClock.Instant? = nil
                var stoppedOnEos = false
                let t0 = SuspendingClock.now
                for try await out in stream {
                    if firstAt == nil { firstAt = .now }
                    if let e = eosId, Int(out.tokenId) == e { stoppedOnEos = true; break }
                    generated.append(Int(out.tokenId))
                }
                let t1 = SuspendingClock.now
                let ttft = firstAt.map { Bench.seconds(from: t0, to: $0) } ?? 0
                let genS = firstAt.map { Bench.seconds(from: $0, to: t1) } ?? 0
                let tps = generated.count > 1 && genS > 0 ? Double(generated.count - 1) / genS : 0
                let capped = !stoppedOnEos && generated.count >= maxTokens
                let text = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                let rec: [String: Any] = ["i": item.i, "prompt_ids": ids.count, "tokens": generated.count, "eos": stoppedOnEos,
                                          "capped": capped, "ttft_s": (ttft * 1000).rounded() / 1000,
                                          "gen_s": (genS * 1000).rounded() / 1000, "gen_tps": (tps * 100).rounded() / 100,
                                          "gold": item.gold, "text": text]
                if let d = try? JSONSerialization.data(withJSONObject: rec), let s = String(data: d, encoding: .utf8) {
                    answers.write(s)
                }
                done += 1
                log.add(String(format: "TASK i=%d prompt=%d tokens=%d ttft_s=%.2f gen_tps=%.2f eos=%d capped=%d elapsed_s=%.0f",
                               item.i, ids.count, generated.count, ttft, tps, stoppedOnEos ? 1 : 0, capped ? 1 : 0,
                               Bench.seconds(from: runStart, to: .now)))
              } catch {
                log.add("TASK i=\(item.i) ERROR \(error) — recreating the engine and continuing")
                if let e = try? await EngineFactory.createEngine(config: configData, modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)) {
                    engine = e
                }
              }
            }
            log.add(String(format: "TASK_DONE task=%@ model=%@ n=%d elapsed_s=%.0f footprint_gb=%.2f thermal=%@",
                           stem, bundle.name, done, Bench.seconds(from: runStart, to: .now),
                           Double(Bench.currentPhysFootprint()) / 1_073_741_824.0, Sustain.thermalName(ProcessInfo.processInfo.thermalState)))
            _ = engine
        } catch {
            log.add("FATAL \(error)")
        }
    }
}
