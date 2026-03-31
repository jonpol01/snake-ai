#set document(title: "Neuroevolution of Snake-Playing Agents", author: "John Soliva")
#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[Neuroevolution of Snake-Playing Agents:\ A GPU-Accelerated Genetic Algorithm Approach]
  #v(0.5cm)
  #line(length: 40%)
  #v(0.3cm)
  #text(size: 12pt)[John Soliva]
  #v(0.15cm)
  #text(size: 10pt, fill: gray)[March 2026 · Revised March 31, 2026]
  #v(0.1cm)
  #text(size: 8.5pt, fill: rgb("#666"))[_Updated with linear fitness formulation, Gaussian mutation, multi-stage environments, and Rust/wgpu implementation_]
  #v(0.6cm)
]

#block(inset: (x: 1.5cm))[
  *Abstract* --- This writeup presents a neuroevolutionary approach to training autonomous agents to play Snake. We employ a genetic algorithm to evolve 2,000 feedforward neural networks using 24 sensory inputs from eight-directional raycasting, two hidden layers of 16 sigmoid-activated neurons, and four output neurons. A linear-lifetime fitness function with exponential score bonuses overcomes degenerate circling behavior that stalled prior quadratic formulations. GPU compute shaders via wgpu dispatch all forward passes in parallel. A multi-stage environment system (Classic, Warehouse, Mixed) enables domain generalization. Agents achieve scores exceeding 50 food items within 500 generations.

  *Keywords:* neuroevolution, genetic algorithm, neural network, Snake, GPU acceleration, wgpu, Rust
]
#v(0.3cm)

= Introduction

The game of Snake presents a deceptively simple yet computationally interesting challenge for artificial intelligence. An agent must navigate a bounded grid, consuming food items that cause its body to grow, while avoiding collisions with walls and its own body. The game requires both short-term tactical decisions and long-term spatial planning.

Neuroevolution offers an alternative to reinforcement learning: rather than training a single network through backpropagation, we evolve a population of networks through selection, crossover, and mutation. This requires no differentiable loss function, no gradient computation, and no labeled data. The population-based search provides natural exploration, and the algorithm is embarrassingly parallel --- ideal for GPU acceleration.

= Background

== Feedforward Neural Networks

A feedforward neural network computes a function through layers of neurons. For a single neuron with input vector $bold(x)$, weight vector $bold(w)$, and bias $b$:

$ y = sigma(bold(w) dot bold(x) + b) $

where $sigma$ is the logistic sigmoid activation:

$ sigma(z) = 1 / (1 + e^(-z)) $

Bias terms are incorporated by appending a constant 1 to each layer's input, yielding a single weight matrix per layer.

== Genetic Algorithms

Genetic algorithms maintain a population of candidate solutions across generations. Each generation, individuals are evaluated by a fitness function, and a new population is produced through selection, crossover, and mutation #cite(<holland1975>).

== Neuroevolution

Neuroevolution applies evolutionary algorithms to optimize neural network weights #cite(<stanley2002>). We adopt fixed-topology neuroevolution where the weight matrices serve as the genome.

= Methodology

== Game Environment

Snake is played on a $20 times 20$ grid (400 cells). The snake starts at center $(10, 10)$ with length 3, facing upward. Death occurs when the head exits the grid, collides with its body, collides with an obstacle, or exhausts its movement allowance. Starting life is 200 steps (250 on obstacle stages), with $+100$ per food eaten (capped at 500).

== Stage System

Three environments are available:

- *Classic*: Empty grid, no obstacles.
- *Warehouse*: Two horizontal rack obstacles (width 3--5 cells) in upper and lower thirds. Obstacles block vision rays and cause death on collision.
- *Mixed*: Randomly selects Classic or Warehouse each generation, forcing domain generalization #cite(<tobin2017>).

Obstacle detection reuses the body vision channel (24 inputs unchanged), requiring no architecture modification. Stage switching preserves all trained brains.

== Sensory Input: Eight-Direction Vision

The network receives 24 inputs from raycasting in 8 directions (cardinal + diagonal). For each direction $d$:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: (left, center, left),
    [*Input*], [*Value*], [*Description*],
    [$v_(d,0)$ (food)], [$0$ or $1$], [Food visible along ray],
    [$v_(d,1)$ (body)], [$0$ or $1$], [Body or obstacle detected],
    [$v_(d,2)$ (wall)], [$1 slash d_"wall"$], [Inverse distance to boundary/obstacle],
  )
]

Total inputs: $8 times 3 = 24$. Obstacles terminate the ray and set $v_(d,1) = 1$, identically to body segments.

== Neural Network Architecture

#align(center)[
  $"Input"(24) arrow.r "Hidden"(16, sigma) arrow.r "Hidden"(16, sigma) arrow.r "Output"(4, sigma)$
]

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center, center),
    [*Layer*], [*Input*], [*Output*], [*Matrix*], [*Params*],
    [0 (in $arrow.r$ h1)], [25], [16], [$16 times 25$], [400],
    [1 (h1 $arrow.r$ h2)], [17], [16], [$16 times 17$], [272],
    [2 (h2 $arrow.r$ out)], [17], [4], [$4 times 17$], [68],
    [*Total*], [], [], [], [*740*],
  )
]

The direction with maximum output is selected as the next move, subject to the constraint that reversal is prohibited.

== Genetic Algorithm

=== Fitness Function

The fitness function was the single most critical design decision. Our initial formulation used quadratic lifetime:

$ F_"quad" = t^2 dot 2^(min(s, 10)) dot max(s - 9, 1) $

where $t$ is lifetime (steps survived) and $s$ is score (food eaten). This created a _degenerate equilibrium_: a circler surviving 200 steps with $s = 0$ achieved $F = 200^2 = 40,000$, while a food-seeker eating 1 item in 80 steps scored only $F = 80^2 dot 2 = 12,800$. Evolution selected _against_ food-seeking.

The solution: *linear lifetime scaling*:

$ F = t dot 2^(min(s, 10)) dot max(s - 9, 1) $

Now a circler ($t = 200, s = 0$) scores $F = 200$, while a snake eating 2 food in 100 steps scores $F = 100 dot 4 = 400$. Any food-eating behavior dominates pure survival. Scores reached 50+ within 500 generations.

=== Selection

Fitness-proportionate (roulette wheel) selection. Parent $i$'s selection probability:

$ P(i) = F_i / (sum_(j=1)^N F_j) $

Two parents are independently selected per offspring.

=== Crossover

Single-point crossover per weight matrix. A random cut point $(r, c)$ is chosen; weights before the cut (row-major order) come from parent A, the rest from parent B.

=== Mutation

Each weight undergoes Gaussian perturbation with probability $p_"mut" = 0.08$:

$ w'_i = "clamp"(w_i + epsilon, -2, 2), quad epsilon tilde cal(N)(0, 0.25^2) $

This preserves learned patterns unlike random replacement ($w'_i tilde U(-1, 1)$), which proved too destructive for the 740-parameter genome.

=== Elitism

The single best brain (by fitness) across all generations is preserved unmodified in the next generation, guaranteeing monotonically non-decreasing peak performance.

= GPU Acceleration

== Compute Shader

The WGSL compute shader dispatches all 2,000 forward passes in parallel:

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, auto, center, left),
    [*Buffer*], [*Size (floats)*], [*Access*], [*Contents*],
    [weights], [$2000 times 740$], [Read], [All network weights],
    [inputs], [$2000 times 24$], [Read], [Vision vectors],
    [outputs], [$2000 times 4$], [R/W], [Direction probabilities],
  )
]

Workgroup size 64, dispatching $ceil(2000 slash 64) = 32$ workgroups. The shader maps to Metal (macOS), Vulkan (Linux), or Direct3D 12 (Windows) via wgpu.

== CPU Fallback

A sequential Rust fallback is automatically selected when GPU compute is unavailable (e.g., Docker containers).

= Implementation

The system is implemented in Rust. The frontend is embedded into the binary via `include_str!()`, eliminating external file dependencies. The axum web framework serves HTTP/WebSocket. The simulation runs in a dedicated tokio task at ~30 FPS. Docker deployment is provided via multi-stage build.

The browser dashboard renders three visualizations: (1) the game grid with the best alive agent, (2) a neural network diagram with color-coded weights, and (3) a fitness graph. A hardware monitor displays GPU backend, inference latency sparkline, and throughput.

= Results

== Fitness Function Impact

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: (left, center, center),
    [*Formulation*], [*Max score (1500 gens)*], [*Outcome*],
    [$t^2 dot 2^s$ (quadratic)], [3], [Permanent stagnation],
    [$t dot 2^s$ (linear)], [50+], [Rapid learning],
  )
]

The quadratic formulation created an absorbing state where perimeter-circling behavior dominated in selection. Linear scaling eliminated this.

== Mutation Strategy

Random replacement mutation ($w' tilde U(-1, 1)$) repeatedly erased useful weight structures. Gaussian perturbation ($w' = w + epsilon$) preserved learned patterns, producing faster convergence.

== Learning Phases

+ *Random movement*, score 0--1 (gen 1--20)
+ *Wall avoidance* via perimeter following (gen 20--100)
+ *Accidental food acquisition* seeds intentional food-seeking (gen 100--300)
+ *Consistent food acquisition*, scores 20--50+ (gen 300--500+)

== Performance

#align(center)[
  #table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: (left, center, center),
    [*Metric*], [*GPU (M1 Max)*], [*CPU Fallback*],
    [Forward pass (2000 NNs)], [2--7 ms], [5--15 ms],
    [Full generation], [2--4 sec], [5--10 sec],
    [Time to score 50+], [~20 min], [~45 min],
    [Memory], [~30 MB], [~25 MB],
  )
]

= Conclusion

The critical finding is that fitness function design dominates all other hyperparameters: linear vs. quadratic lifetime scaling was the difference between permanent stagnation and rapid learning. Gaussian perturbation mutation further improved convergence. The multi-stage system with domain randomization enables training a single brain across diverse environments.

Future directions include NEAT-style topology evolution #cite(<stanley2019>), convolutional full-grid observation, and co-evolutionary multi-snake competition.

#bibliography("refs.yml")
