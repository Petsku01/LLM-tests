                                   ┌────────────────────────────────────┐
                                   │               You                  │
                                   │   Any phone / lightweight laptop   │
                                   └──────────────┬─────────────────────┘
                                                  │ Wi-Fi or Tailscale
                                                  ▼
                             ---------------------------------------------
                              │            Router                         │
                              └───────┬───────────────┬───────────────────┘
                                  │               │
                                  ▼               ▼
                ┌─────────────────────────────────────┐   ┌──────────────────────────┐
                │           laptop-16GB              │   │     laptop-12GB          │
                │   The strongest one (i7/Ryzen 7)    │   │                          │
                │   Role: Orchestrator + Heavy lifter │   │   Role: Reasoning beast  │
                │   • Open WebUI + LiteLLM + Router   │   │   • Mistral-Nemo 12B     │
                │   • CodeLlama 34B q4_K_M (dev-01)   │   │     (reason-01)          │
                └─────────────────┬───────────────────┘   └────────────┬─────────────┘
                                  │                                    │
                                  ▼                                    ▼
                ┌──────────────────────────────────────────────────────────────┐
                │                       laptop-8GB                            │
                │              The weakest one (still perfectly fine)              │
                │              Role: Fast daily driver + RAG scout                 │
                │              • Llama-3.2 8B Instruct q8_0  (assistant-01)      │
                │              • Llama-3.2 3B Instruct + local RAG (research-01)   │
                └──────────────────────────────────────────────────────────────┘
