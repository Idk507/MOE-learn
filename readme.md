A Mixture of Experts (MoE) replaces dense feed-forward (FFN) blocks with many small FFNs called experts, and a learned router/gating network that sends each token (or example) to only a few experts. That gives huge parameter capacity while keeping per-token compute low — so you can pretrain enormous models quickly — but it adds engineering complexity: memory to store all experts, routing logic, load-balancing, and cross-device communication.

What is a Mixture of Experts?
Think of an MoE layer like a call center with many specialists (experts). For each incoming “call” (an input token’s hidden vector), the router decides which specialists to forward the call to. Only those chosen experts compute and respond; their outputs are combined (often weighted) to produce the layer’s output.

<img width="881" height="425" alt="image" src="https://github.com/user-attachments/assets/b022b257-a915-42a3-addd-68d65d29db9e" />
